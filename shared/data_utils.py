# shared/data_utils.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def load_client_csv(path, batch_size=64, val_split=0.2):
    df = pd.read_csv(path)
    if 'subject' in df.columns and 'activity' in df.columns:
        feature_cols = [c for c in df.columns if c not in ('subject','activity')]
        y_col = 'activity'
    else:
        feature_cols = df.columns[:-1].tolist()
        y_col = df.columns[-1]
    X = df[feature_cols].values.astype('float32')
    y = df[y_col].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n = len(X)
    idx = int(n*(1-val_split))
    if idx <= 0:
        idx = max(1, int(n*0.8))
    X_train, X_val = X[:idx], X[idx:]
    y_train, y_val = y_enc[:idx], y_enc[idx:]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
                            batch_size=256)
    in_dim = X.shape[1]
    num_classes = len(np.unique(y_enc))
    return train_loader, val_loader, in_dim, num_classes, le
