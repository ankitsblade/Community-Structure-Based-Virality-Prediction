# microviral/models_ml.py

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import RANDOM_STATE
from logger import logger


def train_baseline_models(
    labeled_df: pd.DataFrame,
) -> Tuple[float, float, Optional[RandomForestClassifier], List[str]]:
    """
    Train baseline models on cascade + community features.

    Models:
      - Logistic Regression (with StandardScaler)
      - RandomForestClassifier

    Behavior:
      - Uses all numeric features from labeled_df (except obvious IDs / label / n_comments)
      - One-hot encodes 'subreddit' (if present) so we can capture cross-subreddit effects.

    Returns:
      - LogReg ROC-AUC (float)
      - RF ROC-AUC (float)
      - trained RandomForest model (or None if training not possible)
      - list of feature names used (List[str])
    """
    if labeled_df is None or len(labeled_df) == 0:
        logger.warning("[Baseline] Empty labeled_df, skipping training.")
        return 0.5, 0.5, None, []

    df = labeled_df.copy()

    if "label" not in df.columns:
        raise ValueError("[Baseline] 'label' column missing in labeled_df.")

    # ------------------------------------------------------------------
    # Encode target
    # ------------------------------------------------------------------
    y = df["label"].astype(int)

    # Need at least 2 classes to train a classifier
    if y.nunique() < 2:
        logger.warning(
            f"[Baseline] Only one class present in labels: {y.unique().tolist()}. "
            "Skipping training."
        )
        return 0.5, 0.5, None, []

    # ------------------------------------------------------------------
    # One-hot encode subreddit (optional, for cross-subreddit diversity)
    # ------------------------------------------------------------------
    if "subreddit" in df.columns:
        sub_dummies = pd.get_dummies(df["subreddit"], prefix="sub")
        df = pd.concat([df.drop(columns=["subreddit"]), sub_dummies], axis=1)

    # ------------------------------------------------------------------
    # Drop obvious non-feature / ID columns
    # ------------------------------------------------------------------
    drop_cols = [
        "label",
        "submission_id",
        "root_user",
        "root_user_id",
        "author",
        "post_id",
        "n_comments",  # used for labeling, not as a feature
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ------------------------------------------------------------------
    # Keep only numeric columns as features
    # ------------------------------------------------------------------
    X = df.select_dtypes(include=[np.number]).fillna(0.0)

    if X.shape[1] == 0:
        logger.warning("[Baseline] No numeric feature columns found, skipping training.")
        return 0.5, 0.5, None, []

    feature_cols = X.columns.tolist()

    logger.info(
        f"[Baseline] Using {len(feature_cols)} features for training: "
        + ", ".join(feature_cols)
    )
    logger.info(
        f"[Baseline] Dataset shape: {X.shape[0]} rows, {X.shape[1]} features; "
        f"class balance: {y.value_counts().to_dict()}"
    )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    except ValueError as e:
        # This can happen for tiny per-subreddit slices with few minority-class samples
        logger.warning(
            f"[Baseline] train_test_split failed (likely too few samples for stratify): {e}"
        )
        return 0.5, 0.5, None, feature_cols

    # ------------------------------------------------------------------
    # Logistic Regression (with scaling)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    preds_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    auc_logreg = roc_auc_score(y_test, preds_logreg)

    # ------------------------------------------------------------------
    # Random Forest (no scaling)
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, preds_rf)

    logger.info(f"[Baseline] LogReg ROC-AUC: {auc_logreg:.4f}")
    logger.info(f"[Baseline] RandomForest ROC-AUC: {auc_rf:.4f}")

    # ------------------------------------------------------------------
    # Feature importances (for logging & visuals)
    # ------------------------------------------------------------------
    importances = rf.feature_importances_
    logger.info("[Baseline] RandomForest feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        logger.info(f"  {name:30s} {imp:.3f}")

    return auc_logreg, auc_rf, rf, feature_cols


def train_community_only_model(labeled_df: pd.DataFrame) -> None:
    """
    Train a simple community-only Logistic Regression model for interpretability.

    Uses only a small set of community features, e.g.:

        - comm_concentration
        - comm_entropy
        - comm_count
    """
    if labeled_df is None or labeled_df.empty:
        logger.warning("[Community-only] Empty labeled_df, skipping.")
        return

    if "label" not in labeled_df.columns:
        logger.warning("[Community-only] 'label' column missing, skipping.")
        return

    df = labeled_df.copy()

    # Select community features
    comm_feats = ["comm_concentration", "comm_entropy", "comm_count"]
    available_feats = [f for f in comm_feats if f in df.columns]

    if not available_feats:
        logger.warning(
            "[Community-only] None of the community features present in DataFrame."
        )
        return

    X = df[available_feats].astype(float).fillna(0.0)
    y = df["label"].astype(int)

    if y.nunique() < 2:
        logger.warning("[Community-only] Need at least 2 classes to train a classifier.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_scaled, y)

    coefs = logreg.coef_[0]
    logger.info("[Community-only] Logistic Regression coefficients:")
    for name, coef in zip(available_feats, coefs):
        logger.info(f"  {name:20s} {coef:+.4f}")


def inspect_logreg_coeffs_community(labeled_df: pd.DataFrame) -> None:
    """
    Convenience wrapper if you want a separate call to just log the community
    coefficients directly (e.g., after filtering to a subset).
    """
    if labeled_df is None or labeled_df.empty:
        logger.warning("[Community-only inspect] Empty labeled_df, skipping.")
        return

    if "label" not in labeled_df.columns:
        logger.warning("[Community-only inspect] 'label' column missing, skipping.")
        return

    df = labeled_df.copy()

    comm_feats = ["comm_concentration", "comm_entropy", "comm_count"]
    available_feats = [f for f in comm_feats if f in df.columns]

    if not available_feats:
        logger.warning(
            "[Community-only inspect] None of the community features present."
        )
        return

    X = df[available_feats].astype(float).fillna(0.0)
    y = df["label"].astype(int)

    if y.nunique() < 2:
        logger.warning(
            "[Community-only inspect] Not enough classes to inspect coefficients."
        )
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_scaled, y)

    coefs = logreg.coef_[0]

    logger.info("[Community-only inspect] Logistic Regression coefficients:")
    for name, coef in zip(available_feats, coefs):
        logger.info(f"  {name:20s} {coef:+.4f}")
