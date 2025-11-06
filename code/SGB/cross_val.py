import numpy as np


def shuffle_indices(indices, random_state=42):
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    return indices


def kfold(n, n_folds=2, random_state=42):
    indices = np.arange(n, dtype=int)
    indices = shuffle_indices(indices, random_state)

    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[:n % n_folds] += 1
    pos = 0
    folds = []
    for size in fold_sizes:
        valid_id = indices[pos: pos+size]
        train_id = np.concatenate([indices[:pos], indices[pos+size:]], dtype=int)
        folds.append((train_id, valid_id))
        pos += size
    return folds


def cross_val(model_class, X, y, score, n_folds=2, l2=0, shuffle_random_state=42, cv=None, **model_args):
    if cv is None:
        cv = kfold(len(X), n_folds, shuffle_random_state)

    result = []
    for train_id, valid_id in cv:
        X_train, X_valid = X.iloc[train_id], X.iloc[valid_id]
        y_train, y_valid = y.iloc[train_id], y.iloc[valid_id]

        model = model_class(**model_args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        if l2 == 0:
            result.append(score(y_valid, y_pred))
        else:
            result.append(score(y_valid, y_pred) + l2*np.linalg.norm(list(model.get_coefs().values())))
    return result