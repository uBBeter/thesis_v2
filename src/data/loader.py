import json
import gzip
import pandas as pd
from pathlib import Path


def load_amazon(path: str | Path) -> pd.DataFrame:
    """Load Amazon review JSON/JSON.GZ into a DataFrame with columns: user, item, rating, timestamp."""
    path = Path(path)
    records = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append({
                "user": obj.get("reviewerID") or obj.get("user_id"),
                "item": obj.get("asin") or obj.get("item_id"),
                "rating": float(obj.get("overall", obj.get("rating", 1.0))),
                "timestamp": int(obj.get("unixReviewTime", obj.get("timestamp", 0))),
            })
    df = pd.DataFrame(records).dropna(subset=["user", "item"])
    return df


def load_yelp(path: str | Path) -> pd.DataFrame:
    """Load Yelp review JSON into a DataFrame with columns: user, item, rating, timestamp."""
    import time
    from datetime import datetime

    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            date_str = obj.get("date", "")
            try:
                ts = int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                ts = 0
            records.append({
                "user": obj.get("user_id"),
                "item": obj.get("business_id"),
                "rating": float(obj.get("stars", 1.0)),
                "timestamp": ts,
            })
    df = pd.DataFrame(records).dropna(subset=["user", "item"])
    return df
