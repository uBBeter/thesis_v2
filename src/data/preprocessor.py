import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Dataset:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user2idx: dict
    item2idx: dict


def kcore_filter(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    while True:
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index
        filtered = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered
    return df.reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    users = sorted(df["user"].unique())
    items = sorted(df["item"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    df = df.copy()
    df["user_idx"] = df["user"].map(user2idx)
    df["item_idx"] = df["item"].map(item2idx)
    return df, user2idx, item2idx


def leave_one_out_split(
    df: pd.DataFrame,
    n_neg: int = 99,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_items = set(df["item_idx"].unique())

    df = df.sort_values(["user_idx", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []

    for user_idx, group in df.groupby("user_idx"):
        items = group["item_idx"].tolist()
        if len(items) < 3:
            for item in items:
                train_rows.append({"user_idx": user_idx, "item_idx": item})
            continue

        history = set(items)
        candidates = list(all_items - history)

        if len(candidates) < n_neg:
            for item in items:
                train_rows.append({"user_idx": user_idx, "item_idx": item})
            continue

        test_pos = items[-1]
        val_pos = items[-2]
        train_items = items[:-2]

        neg_test = rng.choice(candidates, size=n_neg, replace=False).tolist()
        neg_val = rng.choice(candidates, size=n_neg, replace=False).tolist()

        for item in train_items:
            train_rows.append({"user_idx": user_idx, "item_idx": item})
        val_rows.append({"user_idx": user_idx, "pos_item": val_pos, "neg_items": neg_val})
        test_rows.append({"user_idx": user_idx, "pos_item": test_pos, "neg_items": neg_test})

    train = pd.DataFrame(train_rows)
    val = pd.DataFrame(val_rows)
    test = pd.DataFrame(test_rows)
    return train, val, test


def build_dataset(df: pd.DataFrame, k: int = 10, n_neg: int = 99, seed: int = 42) -> Dataset:
    df = kcore_filter(df, k=k)
    df, user2idx, item2idx = remap_ids(df)
    train, val, test = leave_one_out_split(df, n_neg=n_neg, seed=seed)
    return Dataset(
        train=train,
        val=val,
        test=test,
        n_users=len(user2idx),
        n_items=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
    )
