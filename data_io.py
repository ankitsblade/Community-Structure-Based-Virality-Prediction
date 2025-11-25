# microviral/data_io.py

import os
import pandas as pd

from logger import logger


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_nodes_df(nodes_df: pd.DataFrame, path: str = "data/nodes.parquet"):
    """Save cascades (nodes DataFrame) to disk in parquet format."""
    dirpath = os.path.dirname(path)
    ensure_dir(dirpath)
    nodes_df.to_parquet(path, index=False)
    logger.info(f"Saved nodes_df with {len(nodes_df)} rows to {path}")


def load_nodes_df(path: str = "data/nodes.parquet") -> pd.DataFrame:
    """Load cascades (nodes DataFrame) from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached nodes file found at {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded nodes_df with {len(df)} rows from {path}")
    return df
