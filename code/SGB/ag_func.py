from numba import cuda
import numba
import numpy as np
from .coefs import def_coefs
from math import comb

THREADS_PER_BLOCK = 256
MAX_N = 20
TILE = 1024

@cuda.jit
def calculate_monom(trees, coefs, result, comb_num, bias, k):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_idx = cuda.threadIdx.x
    N, obj_num = trees.shape
    active = idx < comb_num
    
    degs = cuda.local.array(MAX_N, dtype=numba.int8)
    
    # вычисление степеней
    if active:
        if N == 1:
            degs[0] = k
        else:
            current_min = 0
            x = idx
            val_old = -1
            j = N - 1
            for i in range(N):
                max_value = k + i
                for val in range(current_min, max_value + 1):
                    remaining_positions = max_value - val + N - i - 2
                    remaining_elements = N - i - 2
        
                    if remaining_positions < 0 or remaining_elements < 0:
                        combinations = 0
                    else:
                        combinations = 1
                        for num in range(remaining_positions, remaining_positions-remaining_elements, -1):
                            combinations *= num
                        for num in range(1, remaining_elements+1):
                            combinations //= num
                        #combinations = comb(remaining_positions, remaining_elements)
        
                    if x < combinations:
                        degs[j] = val - val_old - 1
                        j -= 1
                        val_old = val
                        current_min = val + 1
                        break
                    else:
                        x -= combinations
            degs[0] = (k + N - 1) - val_old - 1
    
    
    # вычисление монома
    sm_trees = cuda.shared.array((MAX_N, TILE), dtype=np.float16)
    local_coef = coefs[idx + bias] if active else 0
    batch_size = TILE // THREADS_PER_BLOCK

    for tile_start in range(0, obj_num, TILE): # переносим предсказания по тайлам
        cur_tile_size = TILE if (tile_start + TILE <= obj_num) else (obj_num - tile_start)

        # Копируем кусок trees[:, tile_start:tile_start+cur_tile_size] -> shared
        for tree_num in range(N):
            for idx_inside_batch in range(batch_size):
                local_obj_idx = batch_size*local_idx + idx_inside_batch
                if local_obj_idx < cur_tile_size:
                    sm_trees[tree_num][local_obj_idx] = trees[tree_num][tile_start + local_obj_idx]
                else:
                    break
        cuda.syncthreads()

        #Считаем моном для каждого объекта из тайла
        if active:
            for i in range(cur_tile_size):
                monom = local_coef
                for tree_num in range(N):
                    monom *= sm_trees[tree_num][i]**degs[tree_num]
                cuda.atomic.add(result, tile_start + i, monom)
        cuda.syncthreads()


def calculate_differential(trees, k, coefs_device, bias):
    N, obj_num = trees.shape
    result = np.zeros(obj_num)

    # вычисление кол-ва комбинаций степеней
    comb_num = comb(N+k-1, N-1)

    # вызов cuda
    result_device = cuda.to_device(result)    # float
    trees_device = cuda.to_device(trees.astype(np.float16))      # float

    blocks_per_grid = (comb_num + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    calculate_monom[blocks_per_grid, THREADS_PER_BLOCK](trees_device, coefs_device, result_device, comb_num, bias, k)

    result = result_device.copy_to_host()

    del result_device, trees_device
    cuda.current_context().deallocations.clear()
    
    return result

def get_bias(start_dif, k, deg_to_add, N):
    bias = 0
    if deg_to_add > 0:
        bias = start_dif[k+deg_to_add]
        for delta in range(deg_to_add):
            bias += comb(N + (k + deg_to_add - delta) - 1, N - 1)
    else:
        bias = start_dif[k]
    return bias

def F(trees, dif_nums=[1], coefs_device=None, start_dif=[-1, 0], deg_to_add=-1):
    if trees.shape[0] == 0:
        return np.zeros(trees.shape[1])

    result = 0
    for dif_num in dif_nums:
        bias = get_bias(start_dif, dif_num, deg_to_add, trees.shape[0])
        sm_mon = calculate_differential(trees, dif_num, coefs_device, bias)
        result += sm_mon
    return result