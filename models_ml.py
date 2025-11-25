# microviral/models_ml.py

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import RANDOM_STATE
from logger import logger


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def prepare_X_y(labeled_df: pd.DataFrame, feature_cols: List[str]):
    """Build feature matrix X and label vector y for given feature columns."""
    X = labeled_df[feature_cols].fillna(0.0)
    y = labeled_df["label"].values
    return X, y


# ------------------------------------------------------------
# 1. Main baseline model (community + dynamics features)
# ------------------------------------------------------------

def train_baseline_models(
    labeled_df: pd.DataFrame,
) -> Tuple[float, float, RandomForestClassifier, List[str]]:
    """
    Train logistic regression and random forest on the full feature set
    (community + early cascade dynamics).
    Returns:
      - LogReg ROC-AUC
      - RF ROC-AUC
      - trained RandomForest model
      - list of feature names
    """

    feature_cols = [
        "comm_concentration",
        "comm_entropy",
        "comm_count",
        "branching_factor",
        "time_to_5",
    ]

    X, y = prepare_X_y(labeled_df, feature_cols)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Logistic Regression (with scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    preds_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    auc_logreg = roc_auc_score(y_test, preds_logreg)

    # Random Forest (no scaling needed)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, preds_rf)

    logger.info(f"[Baseline] LogReg ROC-AUC: {auc_logreg:.4f}")
    logger.info(f"[Baseline] RandomForest ROC-AUC: {auc_rf:.4f}")

    importances = rf.feature_importances_
    logger.info("[Baseline] RandomForest feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        logger.info(f"  {name:20s} {imp:.3f}")

    return auc_logreg, auc_rf, rf, feature_cols


# ------------------------------------------------------------
# 2. Community-only model (for clean community effect story)
# ------------------------------------------------------------

def train_community_only_model(
    labeled_df: pd.DataFrame,
) -> Tuple[float, float, RandomForestClassifier, List[str]]:
    """
    Train Logistic Regression and RandomForest using ONLY community features:
      - comm_concentration
      - comm_entropy
      - comm_count

    This lets you isolate the predictive power of community structure alone.
    """

    feature_cols = [
        "comm_concentration",
        "comm_entropy",
        "comm_count",
    ]

    X, y = prepare_X_y(labeled_df, feature_cols)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    preds_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    auc_logreg = roc_auc_score(y_test, preds_logreg)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, preds_rf)

    logger.info(f"[Community-only] LogReg ROC-AUC: {auc_logreg:.4f}")
    logger.info(f"[Community-only] RandomForest ROC-AUC: {auc_rf:.4f}")

    importances = rf.feature_importances_
    logger.info("[Community-only] RandomForest feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        logger.info(f"  {name:20s} {imp:.3f}")

    return auc_logreg, auc_rf, rf, feature_cols


# ------------------------------------------------------------
# 3. Logistic Regression coefficients for direction of effect
# ------------------------------------------------------------

def inspect_logreg_coeffs_community(
    labeled_df: pd.DataFrame,
) -> None:
    """
    Fit a Logistic Regression on community features only and log
    the coefficients (direction + magnitude).

    Positive coefficient -> feature increases probability of VIRAL (label=1)
    Negative coefficient -> feature pushes towards MICRO-VIRAL (label=0)
    """

    feature_cols = [
        "comm_concentration",
        "comm_entropy",
        "comm_count",
    ]

    X, y = prepare_X_y(labeled_df, feature_cols)

    if len(np.unique(y)) < 2:
        logger.warning("Not enough classes to inspect coefficients.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_scaled, y)

    coefs = logreg.coef_[0]

    logger.info("[Community-only] Logistic Regression coefficients:")
    for name, coef in zip(feature_cols, coefs):
        logger.info(f"  {name:20s} {coef:+.4f}")
