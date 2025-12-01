import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

plt.style.use("seaborn-v0_8")


def ensure_dir(path="figs"):
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# BASIC DISTRIBUTIONS
# ============================================================
def plot_cascade_size_distribution(df, out_dir="figs"):
    """
    Distribution of cascade sizes (submission_id groups).
    Only uses rows where is_submission == True.
    """
    ensure_dir(out_dir)
    sizes = df[df["is_submission"]].groupby("submission_id").size()
    plt.figure(figsize=(7, 5))
    sns.histplot(sizes, bins=40, kde=True)
    plt.title("Cascade Size Distribution (All Subreddits)")
    plt.xlabel("Cascade Size (#nodes)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cascade_size_distribution.png", dpi=300)
    plt.close()


def plot_depth_distribution(df, out_dir="figs"):
    """
    Distribution of node depths within cascades.
    """
    ensure_dir(out_dir)
    if "depth" not in df.columns:
        return
    plt.figure(figsize=(7, 5))
    sns.histplot(df["depth"].dropna(), bins=25, kde=False)
    plt.title("Depth Distribution of Cascade Nodes")
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/depth_distribution.png", dpi=300)
    plt.close()


def plot_subreddit_counts(df, out_dir="figs"):
    """
    Bar chart of number of submissions per subreddit.
    """
    ensure_dir(out_dir)
    if "subreddit" not in df.columns or "is_submission" not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    df[df["is_submission"]]["subreddit"].value_counts().plot(kind="bar")
    plt.title("Number of Posts per Subreddit")
    plt.xlabel("Subreddit")
    plt.ylabel("Posts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/subreddit_counts.png", dpi=300)
    plt.close()


# ============================================================
# FEATURE IMPORTANCE (CLASSICAL ML)
# ============================================================
def plot_feature_importance_global(feature_names, importances, out_dir="figs"):
    """
    Global feature importance (e.g., from a RandomForest) across all subreddits.
    """
    ensure_dir(out_dir)
    feature_names = np.array(feature_names)
    importances = np.array(importances)
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        y=feature_names[order],
        x=importances[order],
        color="skyblue",
    )
    plt.title("Global Feature Importances (All Subreddits)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_global.png", dpi=300)
    plt.close()


def plot_feature_importance_per_subreddit(df_imp_sub, out_dir="figs"):
    """
    Per-subreddit feature importance.
    Expects df_imp_sub with columns: ['subreddit', 'feature', 'importance'].
    """
    ensure_dir(out_dir)
    if df_imp_sub.empty:
        return

    for sub, sdf in df_imp_sub.groupby("subreddit"):
        sdf = sdf.sort_values("importance", ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(y="feature", x="importance", data=sdf, color="salmon")
        plt.title(f"Feature Importance – r/{sub}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/feature_importance_{sub}.png", dpi=300)
        plt.close()


def plot_feature_importance_heatmap(df_imp_sub, out_dir="figs"):
    """
    Optional: heatmap of feature importance (features x subreddits).
    """
    ensure_dir(out_dir)
    if df_imp_sub.empty:
        return

    pivot = df_imp_sub.pivot_table(
        index="feature",
        columns="subreddit",
        values="importance",
        aggfunc="mean",
        fill_value=0.0,
    )
    plt.figure(figsize=(max(8, 0.5 * pivot.shape[1]), 8))
    sns.heatmap(pivot, annot=False, cmap="viridis")
    plt.title("Feature Importance Heatmap (Feature × Subreddit)")
    plt.xlabel("Subreddit")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_heatmap.png", dpi=300)
    plt.close()


# ============================================================
# COMMUNITY & MODULARITY
# ============================================================
def plot_modularity_distribution(mod_df, out_dir="figs"):
    """
    Plot distribution of modularity scores.
    Expects columns: ['modularity'] and optionally ['source'].
    """
    ensure_dir(out_dir)
    if "modularity" not in mod_df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.histplot(mod_df["modularity"], bins=10, kde=True)
    plt.title("User Graph Modularity Distribution")
    plt.xlabel("Modularity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/modularity_distribution.png", dpi=300)
    plt.close()


# ============================================================
# GNN TRAINING PLOTS
# ============================================================
def plot_gnn_training(losses, val_losses, out_dir="figs", tag="global"):
    """
    Simple train vs. val loss curve.
    """
    ensure_dir(out_dir)
    if not losses or not val_losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.title(f"GNN Training Curve ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gnn_training_curve_{tag}.png", dpi=300)
    plt.close()


def plot_gnn_training_auc(train_aucs, val_aucs, out_dir="figs", tag="global"):
    """
    Train vs. val ROC-AUC curves over epochs.
    """
    ensure_dir(out_dir)
    if not train_aucs or not val_aucs:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_aucs) + 1), train_aucs, label="Train AUC")
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, label="Validation AUC")
    plt.title(f"GNN ROC-AUC over Epochs ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gnn_auc_curve_{tag}.png", dpi=300)
    plt.close()


def plot_gnn_metrics(metrics_dict, out_dir="figs", tag="global"):
    """
    Bar plot of summary GNN metrics (train/val/test).
    metrics_dict: e.g. {'train_auc': 0.8, 'val_auc': 0.75, 'test_auc': 0.72}
    """
    ensure_dir(out_dir)
    if not metrics_dict:
        return

    keys = list(metrics_dict.keys())
    vals = [metrics_dict[k] for k in keys]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=keys, y=vals, color="orchid")
    plt.title(f"GNN Evaluation Metrics ({tag})")
    plt.ylabel("Value")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gnn_metrics_{tag}.png", dpi=300)
    plt.close()


def plot_per_subreddit_gnn_metrics(per_sub_metrics, metric="test_auc", out_dir="figs"):
    """
    Per-subreddit bar chart of a chosen metric (e.g., test_auc).
    per_sub_metrics: {subreddit: {train_auc, val_auc, test_auc, ...}}
    """
    ensure_dir(out_dir)
    subs = []
    vals = []

    for sub, m in per_sub_metrics.items():
        if isinstance(m, dict) and metric in m and isinstance(m[metric], (int, float, np.floating)):
            subs.append(sub)
            vals.append(m[metric])

    if not subs:
        return

    plt.figure(figsize=(max(8, 0.4 * len(subs)), 5))
    sns.barplot(x=subs, y=vals)
    plt.title(f"GNN {metric} per Subreddit")
    plt.xlabel("Subreddit")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gnn_{metric}_per_subreddit.png", dpi=300)
    plt.close()


def plot_gnn_ablation_results(ablation_metrics, metric="test_auc", out_dir="figs"):
    """
    Compare feature configs ('full', 'structure_only', etc.) on a chosen metric.
    ablation_metrics: {config_name: {train_auc, val_auc, test_auc, ...}}
    """
    ensure_dir(out_dir)
    configs = []
    vals = []
    for cfg, m in ablation_metrics.items():
        if isinstance(m, dict) and metric in m and isinstance(m[metric], (int, float, np.floating)):
            configs.append(cfg)
            vals.append(m[metric])

    if not configs:
        return

    plt.figure(figsize=(7, 5))
    sns.barplot(x=configs, y=vals)
    plt.title(f"GNN Ablation ({metric} by Feature Config)")
    plt.xlabel("Feature Config")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gnn_ablation_{metric}.png", dpi=300)
    plt.close()
