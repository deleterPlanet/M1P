from numba import cuda
import numpy as np

@cuda.jit
def cuda_newton_root(polys, deriv_polys, x0, tol, max_iter, roots, logs):
    idx = cuda.grid(1)
    if idx >= len(roots):
        return

    for i in range(logs.shape[1]):
        logs[idx][i] = deriv_polys[idx][i]

    for _ in range(max_iter):
        # Вычисляем значение многочлена и его производной в точке x0
        f_x0 = 0
        for coef in polys[idx]:
            f_x0 = f_x0 * x0[idx] + coef

        f_prime_x0 = 0
        for coef in deriv_polys[idx]:
            f_prime_x0 = f_prime_x0 * x0[idx] + coef
        
        # Проверка, чтобы не делить на ноль
        if abs(f_prime_x0) < 1e-12:
            roots[idx] = x0[idx]
            return
        
        # Обновляем приближение
        x1 = x0[idx] - f_x0 / f_prime_x0
        
        # Проверяем сходимость
        if abs(x1 - x0[idx]) < tol:
            break
        x0[idx] = x1
    roots[idx] = x0[idx]

def find_roots(polys, x0=0, tol=1e-7, max_iter=1000):
    roots = np.zeros(polys.shape[0], dtype=np.float64)
    x0_list = np.full(polys.shape[0], x0)
    logs = np.zeros((polys.shape[0], 20))
    # Вычисление производных
    row = np.arange(polys.shape[1] - 1, 0, -1)
    deriv_polys = polys[:, :-1] * row
    
    polys_device = cuda.to_device(polys.astype(np.float64))
    deriv_polys_device = cuda.to_device(deriv_polys.astype(np.float64))
    roots_device = cuda.to_device(roots.astype(np.float64))
    x0_device = cuda.to_device(x0_list.astype(np.float64))
    logs_device = cuda.to_device(logs)

    threads_per_block = 1024
    blocks_per_grid = (polys.shape[0] + threads_per_block - 1) // threads_per_block
    cuda_newton_root[blocks_per_grid, threads_per_block](polys_device, deriv_polys_device, x0_device, tol, max_iter, roots_device, logs_device)

    roots = roots_device.copy_to_host()
    '''
    # print logs
    logs = logs_device.copy_to_host()
    print('logs: ', logs)
    '''
    return roots