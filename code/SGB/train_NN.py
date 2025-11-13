import numpy as np
import pandas as pd
import torch
import os
import time


MAX_FEATURES_NUM = 500

train_tasks = [
    {'name': 'insurance.csv',                      'target': 'charges',                        'features_start': 0},
    {'name': 'task18.xls',                         'target': 'Y -Melting point, K',            'features_start': 2},
    {'name': 'task23.xls',                         'target': 'Tm,K',                           'features_start': 2},
    {'name': '2018Floor1.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor2.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor3.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor4.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor5.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor6.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor7.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': '2018Floor8.csv',                     'target': 'z4_Light(kW)',                   'features_start': 2},
    {'name': 'AirfoilSelfNoise.csv',               'target': 'SSPL',                           'features_start': 0},
    {'name': 'Concrete Compressive Strength.csv',  'target': 'Concrete compressive strength ', 'features_start': 0},
    {'name': 'diabetes.csv',                       'target': 'Outcome',                        'features_start': 0},
    {'name': 'ENB2012_data.csv',                   'target': 'Y2',                             'features_start': 1},
    {'name': 'Folds5x1_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x2_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x3_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x4_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x5_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x6_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x7_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x8_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'Folds5x9_pp.csv',                    'target': 'PE',                             'features_start': 0},
    {'name': 'gt_full1.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full2.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full3.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full4.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full5.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full6.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full7.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full8.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full9.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'gt_full10.csv',                       'target': 'NOX',                            'features_start': 1},
    {'name': 'spotify_songs.csv',                  'target': 'track_popularity',               'features_start': 1},
    {'name': 'task27.xls',                         'target': 'Tm,K',                           'features_start': 2},
    {'name': 'task28.xls',                         'target': 'Tm,K',                           'features_start': 2},
    {'name': 'task31.xls',                         'target': 'Tm, K',                          'features_start': 2},
    {'name': 'task5.xls',                          'target': 'price',                          'features_start': 1},
    {'name': 'task9.xls',                          'target': 'sys',                            'features_start': 2},
    {'name': 'UCI_Real_Estate_Valuation.xlsx',     'target': 'Y house price of unit area',     'features_start': 1},
    {'name': 'winequality-red.csv',                'target': 'quality',                        'features_start': 0},
    {'name': 'avocado1.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado2.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado3.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado4.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado5.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado6.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'avocado7.csv',                       'target': 'AveragePrice',                   'features_start': 2},
    {'name': 'car_price_prediction1.csv',          'target': 'Price',                          'features_start': 2},
    {'name': 'car_price_prediction2.csv',          'target': 'Price',                          'features_start': 2},
    {'name': 'car_price_prediction3.csv',          'target': 'Price',                          'features_start': 2},
    {'name': 'car_price_prediction4.csv',          'target': 'Price',                          'features_start': 2},
    {'name': 'diamonds1.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'diamonds2.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'diamonds3.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'diamonds4.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'diamonds5.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'diamonds6.csv',                      'target': 'price',                          'features_start': 1},
    {'name': 'Laptop_price.csv',                   'target': 'Price',                          'features_start': 1},
    {'name': 'FINAL_USO.csv',                      'target': 'USO_Volume',                     'features_start': 1},
    {'name': 'NFLX.csv',                           'target': 'Volume',                         'features_start': 1},
    {'name': 'retail_price.csv',                   'target': 'lag_price',                      'features_start': 3},
    {'name': 'TSLA.csv',                           'target': 'Volume',                         'features_start': 1},
]


def load_task_dataframe(path):
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in ('.xls', '.xlsx', '.xlt'):
        return pd.read_excel(path)
    elif ext in ('.csv',):
        return pd.read_csv(path)
    else:
        raise ValueError(f'Unsupported extension: {ext} for file {path}')


def collate_fullbatch(batch):
    X = np.stack([b[0] for b in batch], axis=0)
    y = np.stack([b[1] for b in batch], axis=0)
    
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return X_t, y_t


def prepare_xy_from_df(df: pd.DataFrame, target_col: str, features_start: int):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataframe columns")
    y = df[target_col].values
    X = df.iloc[:, features_start:].values
    # drop rows with NaN in target or all-NaN in features
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    # drop rows with NaNs in features
    row_mask = ~np.any(np.isnan(X), axis=1)
    X = X[row_mask]
    y = y[row_mask]
    return X, y


def pad_features(X):
    cur_dim = X.shape[1]
    if cur_dim > MAX_FEATURES_NUM:
        raise ValueError(f"Dataset has {cur_dim} features, but MAX_FEATURES_NUM={MAX_FEATURES_NUM}. Increase MAX_FEATURES_NUM.")
    if cur_dim == MAX_FEATURES_NUM:
        return X
    pad_width = MAX_FEATURES_NUM - cur_dim
    return np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])


def save_checkpoint(model, optimizer, epoch, task_idx, path="checkpoints"):
    # save model state
    model_filename = os.path.join(path, f"checkpoint_epoch_{epoch}_{task_idx}.pt")
    torch.save({
        "epoch": epoch,
        "task_idx": task_idx,
        "model_state": model.state_dict()
    }, model_filename)

    # save full state
    full_filename = os.path.join(path, "full_last_checkpoint.pt")
    torch.save({
        "epoch": epoch,
        "task_idx": task_idx,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, full_filename)

    print(f"[+] Checkpoint saved: {model_filename}")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[+] Restored from: {checkpoint_path}")
    return checkpoint["epoch"], checkpoint["task_idx"]


def train_multi_task(model,
                     tasks_list,
                     data_dir,
                     save_dir,
                     operator_callable,
                     device,
                     epochs=5,
                     lr=1e-4,
                     FGBReg_args={},
                     load_path=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Preloading data
    loaded_tasks = []
    for task_meta in tasks_list:
        df = load_task_dataframe(os.path.join(data_dir, task_meta['name']))
        X, y = prepare_xy_from_df(df, task_meta['target'], task_meta['features_start'])
        X = pad_features(X)
        loaded_tasks.append((task_meta['name'], X, y))
        print(f"Loaded {task_meta['name']}: X={X.shape}, y={y.shape}")

    start_epoch = 0
    if load_path:
        start_epoch, start_task_idx = load_checkpoint(model, optimizer, load_path)
        if start_task_idx == len(train_tasks)-1:
            start_epoch += 1
            start_task_idx = 0
        else:
            start_task_idx += 1

    for epoch in range(start_epoch, start_epoch+epochs):
        print(f"\n===== EPOCH {epoch+1}/{start_epoch+epochs} =====")
        epoch_start = time.time()

        for task_idx, (name, X_np, y_np) in enumerate(loaded_tasks):
            if epoch == start_epoch and task_idx < start_task_idx:
                continue
            print(f"--- Training on {name} ---")

            # Create operator_args
            operator_args = {
                'metric': FGBReg_args.get("metric", None),
                'X_train': X_np,
                'y_train': y_np,
                'FGBReg_args': FGBReg_args,
                'n_folds': 3,
            }

            # Get data
            X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
            y_t = torch.tensor(y_np, dtype=torch.float32, device=device)

            model.train()
            optimizer.zero_grad()

            output, _ = model(X_t, y_t, operator=operator_callable, operator_args=operator_args)

            loss = output if output.dim() == 0 else output.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            save_checkpoint(model, optimizer, epoch, task_idx, save_dir)

            print(f"Loss: {loss.item():.6f}")

        epoch_time = time.time() - epoch_start
        epoch_h = int(epoch_time // 3600)
        epoch_m = int((epoch_time % 3600) // 60)
        epoch_s = int(epoch_time % 60)
        print(f"Epoch time: {epoch_h}h {epoch_m}m {epoch_s}s")

    print("\nTraining finished.")