import numpy as np
from .coefs import def_coefs
from .ag_func import F
from .newton import find_roots
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from math import comb
from numba import cuda
import itertools

def def_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class FGBReg:
    def __init__(
        self, n_estimators=1, M=1, delta=1, theta=0, pred_strategy='deg_root',
        is_normalize=True, F_coefs=None, metric=def_metric, max_depth=None,
        random_state=42
    ):
        self.delta = delta
        self.theta = theta
        self.n_estimators = n_estimators
        self.dif_nums = np.arange(1, M+1)
        self.M = M
        self.pred_strategy = pred_strategy
        self.is_normalize = is_normalize
        self.F_coefs = F_coefs if F_coefs is not None else {}
        self.metric = metric
        self.max_depth = max_depth
        self.random_state = random_state
        self.eps = 10**(-4)
        self.coefs_device = None
        self.coefs_list = None
        self.C_list = None

    def fit(self, X, y, trace=False, X_val=None, y_val=None):
        history = {'train': None, 'val': None}
        obj_num = X.shape[0]
        
        # Нормализация таргета
        if self.is_normalize:
            self.mean = np.mean(y)
            self.std = np.std(y) + self.eps
            y = (y - self.mean) / self.std

        # Подготовка коэффициентов
        self.coefs_list = []
        for key, val in self.F_coefs.items():
            self.coefs_list.append(val)
        self.coefs_device = cuda.to_device(self.coefs_list)

        # Подготовка start_dif
        self.start_dif = [-1, 0]
        for k in range(1, self.dif_nums[-1] + 1):
            N = self.n_estimators
            self.start_dif.append(self.start_dif[-1] + comb(N+k-1, N-1))

        # Генерация базового алгоритма
        polys = np.ones((obj_num, self.dif_nums[-1]+1))
        for deg in range(1, self.dif_nums[-1]):
            short_degs = (deg,)
            sample_deg = list(self.F_coefs.keys())[0]
            true_degs = short_degs + tuple([0]*(len(sample_deg) - len(short_degs)))
            polys[:, -(deg+1)] = self.F_coefs[true_degs]
        polys[:, -1] = -y
        y_new = find_roots(polys, x0=0)

        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, y_new)
        self.forest = [tree]

        y_pred = (self.predict(X) - self.mean) / self.std
        predictions = np.array([tree.predict(X)])

        if trace:
            history['train'] = [self.metric(y_pred, y)]
            if X_val is not None and y_val is not None:
                history['val'] = [self.metric(self.predict(X_val), y_val)]

        # Обучение остальных деревьев
        for N in range(1, self.n_estimators):
            # Подсчёт коэффициентов Ck
            self.C_list = np.zeros((self.M + 1, obj_num))
            self.C_list[0] = y_pred
            for k in range(1, self.M + 1):
                new_dif_nums = self.dif_nums - np.full_like(self.dif_nums, k, dtype=int)
                new_dif_nums = new_dif_nums[new_dif_nums >= 0]
                deg_to_add = k
                Ck = F(predictions, new_dif_nums, self.coefs_device, self.start_dif, deg_to_add)
                self.C_list[k] = Ck

            # Обучение нового дерева
            y_new = self.get_y_new(y, y_pred, predictions, self.coefs_device, N)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, y_new)
            self.forest.append(tree)

            # Обновление предсказания
            predictions = np.append(predictions, [tree.predict(X)], axis=0)
            pred_add = np.zeros_like(y_pred)
            for k in range(1, self.M + 1):
                pred_add += self.C_list[k] * (predictions[-1]**k)
            y_pred += pred_add

            # Подсчёт метрики
            if trace:
                history['train'].append(self.metric(self.predict(X), y))
                if X_val is not None and y_val is not None:
                    history['val'].append(self.metric(self.predict(X_val), y_val))

        if trace:
            return history

    def predict(self, X, get_trees=-1):
        if get_trees == -1:
            get_trees = len(self.forest)
        obj_num = X.shape[0]
        predictions = np.zeros((get_trees, obj_num))
        for idx, tree in enumerate(self.forest[:get_trees]):
            predictions[idx] = tree.predict(X)
        
        # Проверка и восстановление self.coefs_device, если память сброшена
        if self.coefs_device is None or not cuda.is_cuda_array(self.coefs_device):
            print('Восстановление coefs_device на GPU')
            self.coefs_device = cuda.to_device(self.coefs_list)

        y_pred = F(predictions, self.dif_nums, self.coefs_device, self.start_dif, -1)

        if self.is_normalize:
            y_pred = y_pred * self.std + self.mean

        y_pred = np.nan_to_num(y_pred, nan=self.mean if self.is_normalize else 0)
        y_pred[np.isinf(y_pred)] = self.mean if self.is_normalize else 0

        return y_pred

    def get_metric(self, X, y, get_trees=-1):
        y_pred = self.predict(X, get_trees)
        return self.metric(y_pred, y)

    def get_coefs(self):
        return self.F_coefs

    def get_y_new(self, y_true, y_pred=None, predictions=None, coefs_device=None, N=1):
        if y_pred is None:
            y_pred = np.zeros(y_true.shape)
        if predictions is None:
            predictions = np.zeros((0, y_true.shape[0]))
        if coefs_device is None:
            coefs_device = self.coefs_device

        if self.pred_strategy == 'grad':
            C1 = self.C_list[1]
            y_new = (self.delta / N**self.theta) * C1 * (y_true - y_pred)
        elif self.pred_strategy == 'gess':
            C1 = self.C_list[1]
            C2 = self.C_list[2]
            y_new = [0] * len(y_true)
            for i in range(len(y_true)):
                y_new[i] = (self.delta / N**self.theta) * C1[i] * (y_true[i] - y_pred[i])
                if C1[i]**2 + 2 * C2[i] * (y_pred[i] - y_true[i]) > 0:
                    y_new[i] /= C1[i]**2 + 2 * C2[i] * (y_pred[i] - y_true[i])

        y_new_arr = np.array(y_new)
        cleaned_y_new = np.nan_to_num(y_new_arr, nan=0, posinf=0, neginf=0)
        return cleaned_y_new.tolist()
    

class FGBGDReg(FGBReg):
    def __init__(
        self, alpha=1, max_iter=1000, n_estimators=1, M=1, delta=1, theta=0, pred_strategy='deg_root',
        is_normalize=False, F_coefs=def_coefs, metric=def_metric, max_depth=None, random_state=42
    ):
        super().__init__(
            n_estimators=n_estimators, M=M, delta=delta, theta=theta, pred_strategy=pred_strategy,
            is_normalize=is_normalize, F_coefs=F_coefs, metric=metric, max_depth=max_depth, random_state=random_state
        )
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, trace=False, X_val=None, y_val=None):
        history = super().fit(X, y, True, X_val, y_val)

        if self.is_normalize:
            y = (y - self.mean) / self.std

        N, M = self.n_estimators, X.shape[0]
        degs_list = []
        for k in self.dif_nums:
            for dividers in itertools.combinations(range(k + N - 1), N - 1):
                previous = -1
                degs_monom = []
                # Разделяем единицы между разделителями
                for divider in dividers:
                    degs_monom.append(divider - previous - 1)
                    previous = divider
                degs_monom.append(k + N - 1 - previous - 1)

                degs_list.append(tuple(degs_monom))

        predictions = np.zeros((N, M))
        for idx, tree in enumerate(self.forest):
            predictions[idx] = tree.predict(X)

        X_big = np.zeros((X.shape[0], len(degs_list)))

        for j in range(len(degs_list)): # degs
            new_feature = np.ones(M)
            for tree_idx, deg in enumerate(degs_list[j]):
                new_feature *= predictions[tree_idx] ** deg
            X_big[:, j] = new_feature.reshape((M,))

        self.lin_reg = Ridge(alpha=self.alpha, copy_X=True, max_iter=self.max_iter, tol=0, random_state=self.random_state)
        self.lin_reg.fit(X_big, y)

        coefs_dict = {}
        for i in range(len(degs_list)):
            deg = degs_list[i]
            coefs_dict[deg] = self.lin_reg.coef_[i]

        self.F_coefs = coefs_dict.copy()

        self.predict = self.predict_new

        if trace:
            return history

    def predict_new(self, X):
        N, M = self.n_estimators, X.shape[0]
        degs_list = []
        for k in self.dif_nums:
            for dividers in itertools.combinations(range(k + N - 1), N - 1):
                previous = -1
                degs_monom = []
                # Разделяем единицы между разделителями
                for divider in dividers:
                    degs_monom.append(divider - previous - 1)
                    previous = divider
                degs_monom.append(k + N - 1 - previous - 1)

                degs_list.append(tuple(degs_monom))

        predictions = np.zeros((N, M))
        for idx, tree in enumerate(self.forest):
            predictions[idx] = tree.predict(X)

        X_big = np.zeros((X.shape[0], len(degs_list)))

        for j in range(len(degs_list)): # degs
            new_feature = np.ones(M)
            for tree_idx, deg in enumerate(degs_list[j]):
                new_feature *= predictions[tree_idx] ** deg
            X_big[:, j] = new_feature.reshape((M,))

        y_pred = self.lin_reg.predict(X_big)

        if self.is_normalize:
            return y_pred * self.std + self.mean
        else:
            return y_pred


class GB:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Обучение первого алгоритма
        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        tree.fit(X, y)
        self.trees.append(tree)
        predictions = tree.predict(X)

        for _ in range(1, self.n_estimators):
            residuals = y - predictions
            
            # Обучаем новое дерево на остатках
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, self.learning_rate * residuals)
            self.trees.append(tree)
            
            predictions += tree.predict(X)

    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        
        return predictions
    
    def get_coefs(self):
        return np.ones(self.n_estimators)