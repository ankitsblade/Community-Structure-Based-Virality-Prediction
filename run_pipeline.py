# run_pipeline_bluesky.py
"""
Bluesky classical ML + community analysis pipeline.

High-level steps:
1. Load Bluesky cascade nodes from data/bluesky/nodes_multi.parquet
2. Map primary hashtag -> 'subreddit' (so it plugs into existing tooling)
3. Build global user graph (using your existing graphs.py)
4. Detect communities (Louvain or similar)
5. Attach community IDs back to node table
6. Build cascade/community feature table (using your existing features.py)
7. Add virality labels (using your existing add_labels)
8. Train simple baseline ML models (LogReg + RandomForest) on:
   - Global dataset
   - Per-hashtag ("per-subreddit") subsets
9. Save metrics to results/bluesky/, optionally print summary

This is intentionally self-contained for the ML part, so it doesn’t depend
on your existing models_ml.py / visuals.py. You can swap that in later if
you want tighter integration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
)

from logger import logger
from data_io import load_nodes_df
from graphs import (
    build_global_user_graph,
    detect_communities,
    attach_communities_to_nodes,
)
from features import (
    build_feature_table,
    add_labels,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ensure_bluesky_subreddit_column(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    For Bluesky, we treat the primary hashtag as 'subreddit' so the
    downstream code (graphs, features, etc.) can remain unchanged.

    Expected columns coming from your Bluesky nodes builder:
        - cascade_id
        - root_uri
        - primary_hashtag
        - author_did
        - event_type
        - timestamp
    """
    df = nodes_df.copy()

    if "subreddit" not in df.columns:
        if "primary_hashtag" not in df.columns:
            raise ValueError(
                "Bluesky nodes are missing both 'subreddit' and 'primary_hashtag'. "
                "Make sure your bluesky_nodes pipeline writes 'primary_hashtag'."
            )
        logger.info("Mapping primary_hashtag → subreddit for Bluesky.")
        df["subreddit"] = df["primary_hashtag"]

    return df


def _select_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns for ML.
    We drop obvious non-feature stuff like IDs / labels / group names.
    """
    drop_cols = {"cascade_id", "label", "subreddit", "primary_hashtag", "root_uri"}
    numeric_cols = [
        c
        for c in feature_df.columns
        if c not in drop_cols and np.issubdtype(feature_df[c].dtype, np.number)
    ]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for ML.")

    return numeric_cols


def _train_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train two simple baselines:
    - Logistic Regression (LR)
    - Random Forest (RF)

    Returns metrics for both.
    """

    results = {}

    # Some setups will have 2 classes, others 3 (e.g., low/mid/high virality).
    classes = sorted(y.unique())
    n_classes = len(classes)
    logger.info(f"[{model_name}] Classes: {classes} (n={n_classes})")

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # -----------------------------
    # Logistic Regression pipeline
    # -----------------------------
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500,
                multi_class="auto",
                n_jobs=-1,
            )),
        ]
    )

    lr_pipe.fit(X_train, y_train)
    y_prob_lr = lr_pipe.predict_proba(X_test)
    y_pred_lr = lr_pipe.predict(X_test)

    if n_classes == 2:
        # use prob of positive class
        auc_lr = roc_auc_score(y_test, y_prob_lr[:, 1])
    else:
        auc_lr = roc_auc_score(
            y_test, y_prob_lr, multi_class="ovr"
        )

    results["logreg_auc"] = float(auc_lr)
    results["logreg_report"] = classification_report(
        y_test, y_pred_lr, output_dict=True
    )

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)
    y_pred_rf = rf.predict(X_test)

    if n_classes == 2:
        auc_rf = roc_auc_score(y_test, y_prob_rf[:, 1])
    else:
        auc_rf = roc_auc_score(
            y_test, y_prob_rf, multi_class="ovr"
        )

    results["rf_auc"] = float(auc_rf)
    results["rf_report"] = classification_report(
        y_test, y_pred_rf, output_dict=True
    )

    logger.info(
        f"[{model_name}] LogReg AUC: {auc_lr:.4f} | "
        f"RandomForest AUC: {auc_rf:.4f}"
    )

    return results


def _run_global_ml(feature_df: pd.DataFrame) -> Dict[str, Any]:
    feature_cols = _select_feature_columns(feature_df)
    X = feature_df[feature_cols]
    y = feature_df["label"]

    logger.info(f"Global ML: using {len(feature_cols)} numeric features.")
    return _train_single_model(X, y, model_name="GLOBAL")


def _run_per_subreddit_ml(feature_df: pd.DataFrame, min_cascades: int = 30) -> Dict[str, Any]:
    """
    Run baseline models separately for each subreddit/hashtag,
    but only if there are at least `min_cascades` cascades.
    """
    results = {}
    if "subreddit" not in feature_df.columns:
        logger.warning("No 'subreddit' column in feature_df – skipping per-subreddit ML.")
        return results

    for sub, df_sub in feature_df.groupby("subreddit"):
        if len(df_sub) < min_cascades:
            logger.info(
                f"[{sub}] Skipping per-subreddit ML (only {len(df_sub)} cascades, "
                f"need >= {min_cascades})."
            )
            continue

        logger.info(f"[{sub}] Running per-subreddit ML on {len(df_sub)} cascades.")
        feature_cols = _select_feature_columns(df_sub)
        X_sub = df_sub[feature_cols]
        y_sub = df_sub["label"]

        results[sub] = _train_single_model(
            X_sub,
            y_sub,
            model_name=f"SUBREDDIT::{sub}",
        )

    return results


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bluesky virality pipeline (classical ML + community features)."
    )
    parser.add_argument(
        "--nodes-path",
        type=str,
        default="data/bluesky/nodes_multi.parquet",
        help="Path to Bluesky nodes_multi.parquet",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/bluesky",
        help="Directory to save metrics JSONs.",
    )
    parser.add_argument(
        "--min-cascades-per-tag",
        type=int,
        default=30,
        help="Minimum number of cascades per hashtag to run per-tag models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=== BLUESKY PIPELINE (CLASSICAL ML) START ===")
    logger.info(f"Loading Bluesky nodes from {args.nodes_path!r}")

    if not os.path.exists(args.nodes_path):
        raise FileNotFoundError(
            f"Nodes file not found at {args.nodes_path}. "
            "Run your Bluesky nodes builder first."
        )

    # ------------------------------------------------------------------
    # 1. Load nodes + normalize subreddit / hashtag
    # ------------------------------------------------------------------
    nodes_df = load_nodes_df(args.nodes_path)
    logger.info(f"Loaded {len(nodes_df)} node events.")

    nodes_df = _ensure_bluesky_subreddit_column(nodes_df)

    logger.info(
        f"Subreddit/hashtag distribution: "
        f"{nodes_df['subreddit'].value_counts().to_dict()}"
    )

    # ------------------------------------------------------------------
    # 2. Build global user graph + detect communities
    # ------------------------------------------------------------------
    logger.info("Building global user graph from Bluesky nodes…")
    G = build_global_user_graph(nodes_df)

    logger.info("Detecting communities (Louvain / modularity-based)…")
    partition = detect_communities(G)

    logger.info("Attaching community IDs back to nodes…")
    nodes_with_comm = attach_communities_to_nodes(nodes_df, partition)

    # ------------------------------------------------------------------
    # 3. Build cascade/community features + virality labels
    # ------------------------------------------------------------------
    logger.info("Building cascade + community feature table…")
    feature_df = build_feature_table(nodes_with_comm)
    logger.info(f"Feature table shape: {feature_df.shape}")

    logger.info("Adding virality labels using global quantiles…")
    labeled_df = add_labels(feature_df)

    logger.info(
        "Label distribution (global): "
        f"\n{labeled_df['label'].value_counts()}"
    )

    # ------------------------------------------------------------------
    # 4. Global baseline ML
    # ------------------------------------------------------------------
    logger.info("\n=== GLOBAL BASELINE MODELS (Bluesky) ===")
    global_results = _run_global_ml(labeled_df)

    # ------------------------------------------------------------------
    # 5. Per-hashtag / per-subreddit ML
    # ------------------------------------------------------------------
    logger.info("\n=== PER-HASHTAG BASELINE MODELS (Bluesky) ===")
    per_sub_results = _run_per_subreddit_ml(
        labeled_df,
        min_cascades=args.min_cascades_per_tag,
    )

    # ------------------------------------------------------------------
    # 6. Save metrics
    # ------------------------------------------------------------------
    results_dir = Path(args.results_dir)
    _save_json(global_results, results_dir / "metrics_global.json")
    _save_json(per_sub_results, results_dir / "metrics_per_subreddit.json")

    logger.info(f"Saved global metrics → {results_dir / 'metrics_global.json'}")
    logger.info(f"Saved per-subreddit metrics → {results_dir / 'metrics_per_subreddit.json'}")

    logger.info("=== BLUESKY PIPELINE COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
