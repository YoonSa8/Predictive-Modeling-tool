import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import os


def load_and_preprocess(csv_path, label_column):
    df = pd.read_csv(csv_path)

    # Drop columns with mostly missing values
    df.dropna(axis=1, thresh=int(0.8 * len(df)), inplace=True)

    # Separate label and features
    y = df[label_column]
    x = df.drop(columns=[label_column])

    # Encode all non-numeric (categorical) columns
    for col in x.select_dtypes(include=['object', 'category']).columns:
        x[col] = LabelEncoder().fit_transform(x[col].astype(str))

    # Fill missing numeric values with mean
    x = x.fillna(x.mean(numeric_only=True))

    # Fill any remaining missing values with mode
    x = x.fillna(x.mode(numeric_only=False).iloc[0])

    # Encode target if necessary
    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Convert to tensors
    x = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Standardize features
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    x = (x - mean) / std

    # Train/test split
    indices = torch.randperm(x.size(0))
    train_size = int(0.8 * x.size(0))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def create_dataloaders(x_train, y_train, x_test, y_test, batch_size=32):
    num_workers = 2 if os.name != 'nt' else 0
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    if num_workers > 0:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, prefetch_factor=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=num_workers, prefetch_factor=2)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader
