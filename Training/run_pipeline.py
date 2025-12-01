# run_pipeline.py

"""
Reddit classical ML + community analysis pipeline.

High-level steps:
1. Load Reddit cascade nodes from data/nodes_multi.parquet (or --nodes-path)
2. Build global user graph
3. Detect communities (Louvain or similar)
4. Attach community IDs back to node table
5. Compute global user-graph modularity
6. Build cascade/community feature table
7. Add virality labels (global quantiles)
8. Train simple baseline ML models (LogReg + RandomForest) on:
   - Global dataset
   - Per-subreddit subsets
9. Save metrics to results/reddit/
10. Produce visuals:
    - Basic cascade/subreddit distributions
    - Modularity distribution
    - Global feature importance
    - Per-subreddit feature importance + heatmap
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from networkx.algorithms.community import quality as nx_quality

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
from util.data_io import load_nodes_df
from util.graphs import (
    build_global_user_graph,
    detect_communities,
    attach_communities_to_nodes,
)
from features.features import (
    build_feature_table,
    add_labels,
)
from viz.visuals import (
    plot_cascade_size_distribution,
    plot_depth_distribution,
    plot_subreddit_counts,
    plot_modularity_distribution,
    plot_feature_importance_global,
    plot_feature_importance_per_subreddit,
    plot_feature_importance_heatmap,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _select_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns for ML.
    Drop obvious non-feature columns like IDs / labels / group names.
    """
    drop_cols = {"cascade_id", "label", "subreddit", "root_id", "submission_id"}
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

    Returns metrics for both + fitted RF model + feature names.
    """

    results: Dict[str, Any] = {}

    # Some setups will have 2 classes, others 3 (e.g., low/mid/high virality).
    classes = sorted(y.unique())
    n_classes = len(classes)
    logger.info(f"[{model_name}] Classes: {classes} (n={n_classes})")

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    feature_names = list(X.columns)

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
        # use probability of positive class
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
    results["rf_model"] = rf
    results["feature_names"] = feature_names

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
    Run baseline models separately for each subreddit,
    but only if there are at least `min_cascades` cascades.
    """
    results: Dict[str, Any] = {}
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
# Main
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reddit virality pipeline (classical ML + community features)."
    )
    parser.add_argument(
        "--nodes-path",
        type=str,
        default="data/nodes_multi.parquet",
        help="Path to Reddit nodes_multi.parquet",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/reddit",
        help="Directory to save metrics JSONs.",
    )
    parser.add_argument(
        "--min-cascades-per-subreddit",
        type=int,
        default=30,
        help="Minimum number of cascades per subreddit to run per-subreddit models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=== REDDIT PIPELINE (CLASSICAL ML) START ===")
    logger.info(f"Loading Reddit nodes from {args.nodes_path!r}")

    if not os.path.exists(args.nodes_path):
        raise FileNotFoundError(
            f"Nodes file not found at {args.nodes_path}. "
            "Run your Reddit cascades collector first."
        )

    # ------------------------------------------------------------------
    # 1. Load nodes
    # ------------------------------------------------------------------
    nodes_df = load_nodes_df(args.nodes_path)
    logger.info(f"Loaded {len(nodes_df)} node events.")

    if "subreddit" not in nodes_df.columns:
        raise ValueError("Reddit nodes are missing 'subreddit' column.")

    logger.info(
        f"Subreddit distribution: "
        f"{nodes_df['subreddit'].value_counts().to_dict()}"
    )

    figs_dir = "figs/reddit"
    os.makedirs(figs_dir, exist_ok=True)

    # Basic cascade/subreddit distributions
    try:
        plot_cascade_size_distribution(nodes_df, out_dir=figs_dir)
        plot_depth_distribution(nodes_df, out_dir=figs_dir)
        plot_subreddit_counts(nodes_df, out_dir=figs_dir)
        logger.info(f"Saved basic cascade/subreddit distribution plots to {figs_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate basic distribution plots: {e}")

    # ------------------------------------------------------------------
    # 2. Build global user graph + detect communities + modularity
    # ------------------------------------------------------------------
    logger.info("Building global user graph from Reddit nodes…")
    G = build_global_user_graph(nodes_df)

    logger.info("Detecting communities (Louvain / modularity-based)…")
    partition = detect_communities(G)

    # Compute modularity
    try:
        comm2nodes = defaultdict(set)
        for node, cid in partition.items():
            comm2nodes[cid].add(node)
        communities = list(comm2nodes.values())
        modularity_value = nx_quality.modularity(G, communities)
        logger.info(f"Global user graph modularity (Q) = {modularity_value:.4f}")

        mod_df = pd.DataFrame({"source": ["reddit"], "modularity": [modularity_value]})
        plot_modularity_distribution(mod_df, out_dir=figs_dir)
    except Exception as e:
        logger.warning(f"Failed to compute or plot modularity: {e}")

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
    logger.info("\n=== GLOBAL BASELINE MODELS (Reddit) ===")
    global_results = _run_global_ml(labeled_df)

    # ------------------------------------------------------------------
    # 5. Per-subreddit ML
    # ------------------------------------------------------------------
    logger.info("\n=== PER-SUBREDDIT BASELINE MODELS (Reddit) ===")
    per_sub_results = _run_per_subreddit_ml(
        labeled_df,
        min_cascades=args.min_cascades_per_subreddit,
    )

    # ------------------------------------------------------------------
    # 6. Feature importance visuals
    # ------------------------------------------------------------------
    # Global feature importance (RandomForest)
    global_rf = global_results.get("rf_model")
    global_feat_names = global_results.get("feature_names")
    if global_rf is not None and global_feat_names is not None:
        plot_feature_importance_global(
            global_feat_names,
            global_rf.feature_importances_,
            out_dir=figs_dir,
        )

    # Per-subreddit feature importance
    rows = []
    for sub, res in per_sub_results.items():
        rf_model = res.get("rf_model")
        feat_names = res.get("feature_names")
        if rf_model is None or feat_names is None:
            continue
        importances = rf_model.feature_importances_
        for fname, imp in zip(feat_names, importances):
            rows.append(
                {"subreddit": sub, "feature": fname, "importance": float(imp)}
            )

    if rows:
        df_imp_sub = pd.DataFrame(rows)
        plot_feature_importance_per_subreddit(df_imp_sub, out_dir=figs_dir)
        plot_feature_importance_heatmap(df_imp_sub, out_dir=figs_dir)

        # Optional: save raw per-subreddit importance table
        imp_path = Path(args.results_dir) / "feature_importance_per_subreddit.csv"
        imp_path.parent.mkdir(parents=True, exist_ok=True)
        df_imp_sub.to_csv(imp_path, index=False)
        logger.info(f"Saved per-subreddit feature importance table → {imp_path}")

    # ------------------------------------------------------------------
    # 7. Save metrics (strip models before JSON)
    # ------------------------------------------------------------------
    results_dir = Path(args.results_dir)

    global_results_json = {
        k: v
        for k, v in global_results.items()
        if k not in {"logreg_model", "rf_model"}
    }
    per_sub_results_json = {}
    for sub, res in per_sub_results.items():
        per_sub_results_json[sub] = {
            k: v
            for k, v in res.items()
            if k not in {"logreg_model", "rf_model"}
        }

    _save_json(global_results_json, results_dir / "metrics_global.json")
    _save_json(per_sub_results_json, results_dir / "metrics_per_subreddit.json")

    logger.info(f"Saved global metrics → {results_dir / 'metrics_global.json'}")
    logger.info(f"Saved per-subreddit metrics → {results_dir / 'metrics_per_subreddit.json'}")

    logger.info("=== REDDIT PIPELINE COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
