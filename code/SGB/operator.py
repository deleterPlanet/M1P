import numpy as np
import pandas as pd

from .cross_val import cross_val
from .optimization import FGBReg


def operator(coefs, metric, X_train, y_train, FGBReg_args, n_folds=3):
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train.reshape(-1))
    elif hasattr(y_train, "values"):
        y_train = pd.Series(y_train.values.reshape(-1))
    else:
        y_train = pd.Series(y_train)

    FGBReg_args["F_coefs"] = coefs

    score_list = cross_val(FGBReg, X_train, y_train, score=metric, n_folds=n_folds, l2=0, shuffle_random_state=42, cv=None, **FGBReg_args)

    return np.mean(score_list)