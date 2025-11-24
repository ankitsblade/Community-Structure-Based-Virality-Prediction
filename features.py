# microviral/features.py

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import entropy

from config import EARLY_K, MIN_COMMENTS_FOR_LABEL, MICRO_PERC, VIRAL_PERC


def get_early_adopters(nodes_df: pd.DataFrame, submission_id: str, k: int = EARLY_K) -> List[str]:
    """Return first k DISTINCT commenters by timestamp for one submission."""
    sub_df = nodes_df[
        (nodes_df["submission_id"] == submission_id) &
        (~nodes_df["is_submission"])
    ].sort_values("created_utc")

    early = (
        sub_df.dropna(subset=["author"])
              .drop_duplicates(subset=["author"])
              .head(k)["author"]
              .tolist()
    )
    return early


def compute_community_features(nodes_df: pd.DataFrame, submission_id: str, k: int = EARLY_K) -> Dict[str, Any]:
    sub = nodes_df[nodes_df["submission_id"] == submission_id]

    early_df = (
        sub[~sub["is_submission"]]
        .dropna(subset=["author"])
        .sort_values("created_utc")
        .drop_duplicates(subset=["author"])
        .head(k)
    )

    communities = early_df["community_id"].dropna().values

    if len(communities) == 0:
        return {
            "comm_concentration": np.nan,
            "comm_entropy": np.nan,
            "comm_count": 0,
        }

    uniq, counts = np.unique(communities, return_counts=True)
    conc = counts.max() / counts.sum()
    ent = entropy(counts)

    return {
        "comm_concentration": conc,
        "comm_entropy": ent,
        "comm_count": len(uniq),
    }


def compute_cascade_features(nodes_df: pd.DataFrame, submission_id: str) -> Dict[str, Any]:
    """Early-ish cascade features. Avoid final-size leakage where possible."""
    sub = nodes_df[nodes_df["submission_id"] == submission_id]
    comments = sub[~sub["is_submission"]].sort_values("created_utc")

    if comments.empty:
        return {
            "n_comments": 0,
            "max_depth": 0,
            "branching_factor": 0.0,
            "time_to_5": np.nan,
        }

    n_comments = len(comments)
    max_depth = comments["depth"].max()

    root_mask = comments["parent_id"] == submission_id
    root_comments = root_mask.sum()
    branching = root_comments / n_comments if n_comments > 0 else 0.0

    if n_comments >= 5:
        t0 = comments["created_utc"].iloc[0]
        t5 = comments["created_utc"].iloc[4]
        time_to_5 = t5 - t0
    else:
        time_to_5 = np.nan

    return {
        "n_comments": n_comments,          # used ONLY for labeling
        "max_depth": max_depth,
        "branching_factor": branching,
        "time_to_5": time_to_5,
    }


def build_feature_table(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Compute community + cascade features for every submission."""
    rows = []
    for sub_id in nodes_df["submission_id"].unique():
        comm_feats = compute_community_features(nodes_df, sub_id)
        cas_feats = compute_cascade_features(nodes_df, sub_id)
        row = {
            "submission_id": sub_id,
            **comm_feats,
            **cas_feats,
        }
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    return feature_df


def add_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Label cascades as micro-viral (0) or viral (1) based on n_comments quantiles."""
    df = feature_df.copy()
    df = df[df["n_comments"] >= MIN_COMMENTS_FOR_LABEL]

    sizes = df["n_comments"]
    p_micro = sizes.quantile(MICRO_PERC)
    p_viral = sizes.quantile(VIRAL_PERC)

    def label_row(n):
        if n >= p_viral:
            return 1  # viral
        elif n >= p_micro:
            return 0  # micro-viral
        else:
            return np.nan

    df["label"] = df["n_comments"].apply(label_row)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df
