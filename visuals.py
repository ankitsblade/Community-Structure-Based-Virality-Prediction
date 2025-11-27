# microviral/visuals.py

import os
from typing import List, Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from logger import logger


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _infer_single_subreddit(df: pd.DataFrame) -> Optional[str]:
    """
    If the DataFrame has a 'subreddit' column and only one unique value,
    return that subreddit name; otherwise return None.
    """
    if "subreddit" in df.columns:
        subs = df["subreddit"].dropna().unique()
        if len(subs) == 1:
            return str(subs[0])
    return None


# -------------------------------------------------------
# 1. Cascade size distribution
# -------------------------------------------------------

def plot_cascade_size_distribution(
    feature_df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "cascade_size_distribution.png",
) -> None:
    """
    Plot histogram of cascade sizes (n_comments) across ALL cascades.
    """
    _ensure_dir(output_dir)

    if "n_comments" not in feature_df.columns:
        logger.warning("feature_df has no 'n_comments' column; skipping distribution plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(feature_df["n_comments"], bins=30)
    plt.xlabel("Cascade size (number of comments)")
    plt.ylabel("Frequency")

    sub = _infer_single_subreddit(feature_df)
    if sub is not None:
        title = f"Cascade Size Distribution — r/{sub}"
    else:
        title = "Cascade Size Distribution (all subreddits)"
    plt.title(title)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Cascade size distribution saved to {out_path}")


# -------------------------------------------------------
# 2. Label balance
# -------------------------------------------------------

def plot_label_balance(
    labeled_df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "label_balance.png",
) -> None:
    """
    Plot number of cascades per class (0=micro-viral, 1=viral).
    Automatically annotates subreddit in the title if the DF is single-subreddit.
    """
    _ensure_dir(output_dir)

    if "label" not in labeled_df.columns:
        logger.warning("labeled_df has no 'label' column; skipping label balance plot.")
        return

    counts = labeled_df["label"].value_counts().sort_index()
    labels = ["micro-viral (0)", "viral (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Number of cascades")

    sub = _infer_single_subreddit(labeled_df)
    if sub is not None:
        title = f"Label Balance — r/{sub}"
    else:
        title = "Label Balance (all subreddits)"
    plt.title(title)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Label balance plot saved to {out_path}")


# -------------------------------------------------------
# 3. Feature importances
# -------------------------------------------------------

def plot_feature_importances(
    feature_names: Iterable[str],
    importances: np.ndarray,
    output_dir: str = "figures",
    filename: str = "feature_importances.png",
    title_suffix: Optional[str] = None,
) -> None:
    """
    Plot a bar chart of feature importances (e.g., from RandomForest).

    title_suffix can be used to add something like 'r/politics' or
    'community-only', but the function is fully backward-compatible.
    """
    _ensure_dir(output_dir)

    feature_names = list(feature_names)
    importances = np.asarray(importances)

    if importances.shape[0] != len(feature_names):
        logger.warning("Length mismatch between feature_names and importances; skipping plot.")
        return

    order = np.argsort(-importances)
    names_sorted = [feature_names[i] for i in order]
    imps_sorted = importances[order]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(names_sorted)), imps_sorted)
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=45, ha="right")
    plt.ylabel("Importance")

    title = "Feature Importances"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Feature importances plot saved to {out_path}")


# -------------------------------------------------------
# 4. Feature-by-label (e.g., community features)
# -------------------------------------------------------

def plot_feature_by_label(
    labeled_df: pd.DataFrame,
    feature_name: str,
    output_dir: str = "figures",
    filename_prefix: Optional[str] = None,
) -> None:
    """
    Plot distribution of a single feature split by label (0 vs 1),
    and run a Mann-Whitney U test to check if distributions differ.

    Automatically annotates subreddit in title if DF is single-subreddit.

    filename_prefix: if None → "feature_by_label_<feature_name>"
                     if provided → used as prefix (e.g., "feature_by_label_comm_concentration_politics")
    """
    _ensure_dir(output_dir)

    if feature_name not in labeled_df.columns:
        logger.warning(f"Feature '{feature_name}' not found in labeled_df; skipping.")
        return

    if "label" not in labeled_df.columns:
        logger.warning("labeled_df has no 'label' column; skipping feature-by-label plot.")
        return

    df = labeled_df[[feature_name, "label"]].dropna()
    if df["label"].nunique() < 2:
        logger.warning(f"Not enough classes for feature '{feature_name}' (only one label present).")
        return

    vals_0 = df[df["label"] == 0][feature_name].values
    vals_1 = df[df["label"] == 1][feature_name].values

    if len(vals_0) == 0 or len(vals_1) == 0:
        logger.warning(f"Not enough samples in one of the label groups for '{feature_name}'.")
        return

    # Mann-Whitney U test
    try:
        stat, p_val = mannwhitneyu(vals_0, vals_1, alternative="two-sided")
        logger.info(
            f"[{feature_name}] Mann-Whitney U test: U={stat:.3f}, p={p_val:.3e} "
            "(0=micro-viral, 1=viral)"
        )
    except Exception as e:
        logger.warning(f"Could not compute Mann-Whitney U for {feature_name}: {e}")

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [vals_0, vals_1],
        labels=["micro-viral (0)", "viral (1)"],
        showfliers=True,
    )
    plt.ylabel(feature_name)

    sub = _infer_single_subreddit(labeled_df)
    if sub is not None:
        title = f"{feature_name} by label — r/{sub}"
    else:
        title = f"{feature_name} by label"
    plt.title(title)

    if filename_prefix is None:
        filename_prefix = f"feature_by_label_{feature_name}"

    fname = f"{filename_prefix}.png"
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Feature-by-label plot for '{feature_name}' saved to {out_path}")


# -------------------------------------------------------
# 5. Cascade graph visualization
# -------------------------------------------------------

def plot_cascade_graph(
    nodes_df: pd.DataFrame,
    submission_id: str,
    early_authors: Optional[List[str]] = None,
    max_nodes: int = 250,
    output_dir: str = "figures",
    filename: Optional[str] = None,
) -> None:
    """
    Visualize a single cascade as a graph:

    - Red node: submission (root)
    - Orange nodes: early adopters (first K distinct authors)
    - Skyblue: other commenters

    Automatically includes subreddit in the title if it can be inferred.

    max_nodes: caps the number of nodes for readability by taking earliest nodes in time.
    """
    _ensure_dir(output_dir)

    sub_df = nodes_df[nodes_df["submission_id"] == submission_id].copy()
    if sub_df.empty:
        logger.warning(f"No nodes found for submission_id={submission_id}; skipping cascade plot.")
        return

    # Sort by time and cap
    if "created_utc" in sub_df.columns:
        sub_df = sub_df.sort_values("created_utc")
    if len(sub_df) > max_nodes:
        logger.info(
            f"Trimming cascade for {submission_id} from {len(sub_df)} to {max_nodes} nodes "
            "for visualization."
        )
        # Always keep the submission root, plus the earliest comments
        root = sub_df[sub_df["is_submission"]]
        comments = sub_df[~sub_df["is_submission"]].head(max_nodes - len(root))
        sub_df = pd.concat([root, comments], ignore_index=True)

    # Build DiGraph
    G = nx.DiGraph()

    # Add nodes
    for _, row in sub_df.iterrows():
        nid = row["node_id"]
        G.add_node(
            nid,
            is_submission=bool(row["is_submission"]),
            author=row.get("author", None),
        )

    # Add edges
    for _, row in sub_df.iterrows():
        parent_id = row["parent_id"]
        child_id = row["node_id"]
        if pd.notna(parent_id) and parent_id in G.nodes:
            G.add_edge(parent_id, child_id)

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Colors and sizes
    if early_authors is None:
        early_authors = []
    early_authors_set = set(early_authors)

    node_colors = []
    node_sizes = []
    for n, data in G.nodes(data=True):
        if data.get("is_submission", False):
            node_colors.append("red")
            node_sizes.append(400)
        elif data.get("author") in early_authors_set:
            node_colors.append("orange")
            node_sizes.append(200)
        else:
            node_colors.append("skyblue")
            node_sizes.append(80)

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        arrows=False,
        with_labels=False,
        alpha=0.9,
    )

    # Title with subreddit if possible
    sub = None
    if "subreddit" in sub_df.columns:
        subs = sub_df["subreddit"].dropna().unique()
        if len(subs) == 1:
            sub = subs[0]

    if sub is not None:
        title = f"Cascade Graph — {submission_id} (r/{sub})"
    else:
        title = f"Cascade Graph — {submission_id}"
    plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if filename is None:
        filename = f"cascade_graph_{submission_id}.png"

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Cascade graph for submission_id={submission_id} saved to {out_path}")


# -------------------------------------------------------
# 6. MULTI-SUBREDDIT PLOTS
# -------------------------------------------------------

def plot_label_balance_multi(
    labeled_df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "label_balance_by_subreddit.png",
) -> None:
    """
    One figure that shows label counts (0 / 1) for every subreddit.

    Produces a grouped bar chart:
      x-axis: subreddits
      bars: micro-viral (0), viral (1)
    """
    _ensure_dir(output_dir)

    if "label" not in labeled_df.columns or "subreddit" not in labeled_df.columns:
        logger.warning(
            "Need 'label' and 'subreddit' columns for plot_label_balance_multi; skipping."
        )
        return

    # Build contingency table: rows=subreddits, cols=labels
    table = (
        labeled_df
        .groupby(["subreddit", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    subs = table.index.tolist()
    counts_0 = table.get(0, pd.Series(0, index=subs)).values
    counts_1 = table.get(1, pd.Series(0, index=subs)).values

    x = np.arange(len(subs))
    width = 0.35

    plt.figure(figsize=(1.5 * len(subs) + 4, 5))
    plt.bar(x - width / 2, counts_0, width, label="micro-viral (0)")
    plt.bar(x + width / 2, counts_1, width, label="viral (1)")

    plt.xticks(x, [f"r/{s}" for s in subs], rotation=30, ha="right")
    plt.ylabel("Number of cascades")
    plt.title("Label Balance by Subreddit")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Multi-subreddit label balance plot saved to {out_path}")


def plot_feature_by_label_multi(
    labeled_df: pd.DataFrame,
    feature_name: str,
    output_dir: str = "figures",
    filename: Optional[str] = None,
) -> None:
    """
    One figure per feature that shows boxplots for each subreddit:

      - columns: subreddits
      - within each subplot: boxplot for label 0 vs 1

    So instead of 3 separate PNGs per feature, you get a single
    'feature_by_label_<feature_name>_by_subreddit.png'.
    """
    _ensure_dir(output_dir)

    required_cols = {feature_name, "label", "subreddit"}
    if not required_cols.issubset(labeled_df.columns):
        logger.warning(
            f"Need columns {required_cols} in labeled_df for plot_feature_by_label_multi; skipping."
        )
        return

    subs = sorted(labeled_df["subreddit"].dropna().unique())
    if len(subs) == 0:
        logger.warning("No subreddits found for multi feature-by-label plot; skipping.")
        return

    n_subs = len(subs)
    fig, axes = plt.subplots(
        1, n_subs,
        figsize=(4 * n_subs + 2, 4),
        sharey=True
    )

    if n_subs == 1:
        axes = [axes]

    for ax, sub in zip(axes, subs):
        df_sub = labeled_df[labeled_df["subreddit"] == sub]
        vals_0 = df_sub[df_sub["label"] == 0][feature_name].dropna().values
        vals_1 = df_sub[df_sub["label"] == 1][feature_name].dropna().values

        if len(vals_0) == 0 or len(vals_1) == 0:
            ax.text(0.5, 0.5, "not enough data", ha="center", va="center")
            ax.set_title(f"r/{sub}")
            ax.set_xticks([])
            continue

        ax.boxplot(
            [vals_0, vals_1],
            labels=["0", "1"],
            showfliers=True,
        )
        ax.set_title(f"r/{sub}")
        ax.set_xlabel("label")

    fig.suptitle(f"{feature_name} by label across subreddits")
    axes[0].set_ylabel(feature_name)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if filename is None:
        filename = f"feature_by_label_{feature_name}_by_subreddit.png"

    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Multi-subreddit feature-by-label plot for '{feature_name}' saved to {out_path}")
