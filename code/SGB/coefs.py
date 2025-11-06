import math
from numba import cuda
import numpy as np
import itertools

class step_coefs:
    def __init__(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, degs):
        if sum(degs) == 0:
            return 1
        return self.gamma / (sum(degs) ** self.beta)
    
class changeable_coefs:
    def __init__(self, coefs_dict):
        self.coefs_dict = coefs_dict

    def __call__(self, degs):
        sample_deg = list(self.coefs_dict.keys())[0]
        true_degs = degs + tuple([0]*(len(sample_deg) - len(degs)))
        return self.coefs_dict[true_degs]

def def_coefs(degs):
    return 1.0
    
def factorials_coefs(degs):
    coef = 1
    for deg in degs:
        coef /= math.factorial(deg)
    return coef

def garm_coefs(degs):
    return 1/sum(degs)


def make_coefs_dict_from_func(coefs, M, N):
    coefs_dict = {}
    for k in np.arange(1, M+1):
        for dividers in itertools.combinations(range(k + N - 1), N - 1):
            previous = -1
            degs_monom = []
            # Разделяем единицы между разделителями
            for divider in dividers:
                degs_monom.append(divider - previous - 1)
                previous = divider
            degs_monom.append(k + N - 1 - previous - 1)
            degs_monom = np.array(degs_monom, np.int8)
            coefs_dict[tuple(degs_monom)] = coefs(degs_monom)
    return coefs_dict


def make_coefs_dict_from_list(coefs_list, M, N):
    coefs_dict = {}
    for k in np.arange(1, M+1):
        for dividers in itertools.combinations(range(k + N - 1), N - 1):
            previous = -1
            degs_monom = []
            # Разделяем единицы между разделителями
            for divider in dividers:
                degs_monom.append(divider - previous - 1)
                previous = divider
            degs_monom.append(k + N - 1 - previous - 1)
            degs_monom = np.array(degs_monom, np.int8)
            coefs_dict[tuple(degs_monom)] = 0

    for idx, key in enumerate(coefs_dict.keys()):
        coefs_dict[key] = coefs_list[idx]
    return coefs_dict