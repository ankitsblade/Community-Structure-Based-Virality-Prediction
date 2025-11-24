# microviral/models_ml.py

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import RANDOM_STATE


def prepare_X_y(labeled_df: pd.DataFrame):
    """Build feature matrix X and label vector y with no leakage."""
    # IMPORTANT: we deliberately exclude n_comments (label driver)
    feature_cols = [
        "comm_concentration",
        "comm_entropy",
        "comm_count",
        "branching_factor",
        "time_to_5",
    ]
    X = labeled_df[feature_cols].fillna(0.0)
    y = labeled_df["label"].values
    return X, y, feature_cols


def train_baseline_models(labeled_df: pd.DataFrame) -> Tuple[float, float]:
    """Train logistic regression and random forest; return their ROC-AUC."""
    X, y, feature_cols = prepare_X_y(labeled_df)

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

    print("LogReg ROC-AUC:", auc_logreg)
    print("RandomForest ROC-AUC:", auc_rf)

    # simple feature importance from RF
    importances = rf.feature_importances_
    print("\nRandomForest feature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {name:20s} {imp:.3f}")

    return auc_logreg, auc_rf
