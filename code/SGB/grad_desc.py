from math import comb
from tqdm import tqdm
import numpy as np
import pickle

from .cross_val import kfold
from .optimization import FGBReg
from .operator import operator


def get_grad_indices(degs, n_coefs_folds):
    M = sum(degs[-1])
    N = len(degs[0])

    start_dif = [0]
    for k in range(1, M + 1):
        start_dif.append(start_dif[-1] + comb(N+k-1, N-1))

    indices_list = []
    for dif_num in range(M):
        if start_dif[dif_num+1] - start_dif[dif_num] <= n_coefs_folds:
            for idx in range(start_dif[dif_num], start_dif[dif_num+1]):
                indices_list.append([idx])
        else:
            folds = kfold(start_dif[dif_num+1] - start_dif[dif_num], n_folds=n_coefs_folds, random_state=randint(1, 1000))
            for _, indices in folds:
                indices_list.append(start_dif[dif_num] + indices)
    return indices_list


def get_test_score(coefs, metric, X_train, y_train, X_test, y_test, FGBReg_args):
    FGBReg_args['F_coefs'] = coefs
    model = FGBReg(**FGBReg_args)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def hyper_train(coefs, X_train, y_train, X_test, y_test, FGBReg_args, metric, phi_0=0.1, beta=0.9, eps=1e-2, epoch_num=3, n_coefs_folds=100, task_name="task0", epoch_start=0, v=None):
    phi_schedule = lambda epoch: phi_0 * (0.9 ** epoch)

    coefs_keys = list(coefs.keys())
    num_coefs = len(coefs_keys)

    if not v:
        v = {deg: 0.0 for deg in coefs_keys}

    indices_list = get_grad_indices(coefs_keys, n_coefs_folds)

    print(f"Start Loss Train: {operator(coefs, metric, X_train, y_train, FGBReg_args, n_folds=10):.4f}",
          f"Loss Test: {operator(coefs, metric, X_test, y_test, FGBReg_args, n_folds=10):.4f}")

    for epoch in range(epoch_start, epoch_start + epoch_num):
        print(f"Start epoch: {epoch}")

        for indices in tqdm(indices_list):
            F_w = operator(coefs, metric, X_train, y_train, FGBReg_args)

            mean_coef = np.mean([coefs[coefs_keys[idx]] for idx in indices])

            # Усреднённая оценка градиента
            eps_values = [eps, -eps] #[eps, -eps, eps/2, -eps/2]
            grads = []

            for e in eps_values:
                copy_coefs = coefs.copy()
                for idx in indices:
                    deg = coefs_keys[idx]
                    copy_coefs[deg] += mean_coef * e

                F_delta_w = operator(copy_coefs, metric, X_train, y_train, FGBReg_args)
                grad_est = (F_delta_w - F_w) / ((mean_coef * e) * (len(indices) ** 1.5))
                grads.append(grad_est)

            grad_F_i = np.mean(grads)

            # Градиентный шаг
            if abs(grad_F_i) <= 2*mean_coef:
                for idx in indices:
                    deg = coefs_keys[idx]
                    v[deg] = beta * v[deg] + (1 - beta) * grad_F_i
                    coefs[deg] -= phi_schedule(epoch) * v[deg]
            '''
            print(
                f"Loss Train: {operator(coefs, metric, X_train, y_train, FGBReg_args):.4f}",
                f"Loss Test: {operator(coefs, metric, X_test, y_test, FGBReg_args):.4f}",
                f"mean_coef: {mean_coef:.4f}",
                f"indoces: {indices}",
            )
            '''

        print(
            f"Loss Train: {operator(coefs, metric, X_train, y_train, FGBReg_args, n_folds=10):.4f}",
            f"Loss Test: {operator(coefs, metric, X_test, y_test, FGBReg_args, n_folds=10):.4f}",
            )

        with open(f'{task_name}_coefs_dict_epoch{epoch+1}.pkl', 'wb') as f:
            pickle.dump(coefs, f)

        with open(f'{task_name}_momentum_dict_epoch{epoch+1}.pkl', 'wb') as f:
            pickle.dump(v, f)