# microviral/visuals.py

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .logger import logger


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Cascade size & label distributions
# ---------------------------------------------------------------------

def plot_cascade_size_distribution(
    feature_df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "cascade_size_distribution.png",
):
    """Plot histogram of cascade sizes (n_comments) and save to file."""
    _ensure_dir(output_dir)

    sizes = feature_df["n_comments"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=30)
    plt.xlabel("Cascade size (number of comments)")
    plt.ylabel("Frequency")
    plt.title("Cascade Size Distribution")

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Cascade size distribution saved to {out_path}")


def plot_label_balance(
    labeled_df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "label_balance.png",
):
    """
    Plot bar chart of label counts (micro-viral vs viral).
    Assumes labels: 0 = micro-viral, 1 = viral.
    """
    _ensure_dir(output_dir)

    counts = labeled_df["label"].value_counts().sort_index()
    labels = ["micro-viral (0)", "viral (1)"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Number of cascades")
    plt.title("Label Balance")

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Label balance figure saved to {out_path}")


# ---------------------------------------------------------------------
# 2. Feature importance bar plot
# ---------------------------------------------------------------------

def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    output_dir: str = "figures",
    filename: str = "feature_importances.png",
):
    """
    Plot bar chart of feature importances.
    Typically used with RandomForest feature_importances_.
    """
    _ensure_dir(output_dir)

    order = np.argsort(-importances)
    sorted_names = [feature_names[i] for i in order]
    sorted_imps = importances[order]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sorted_names)), sorted_imps)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("RandomForest Feature Importances")

    out_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Feature importance figure saved to {out_path}")


# ---------------------------------------------------------------------
# 3. Cascade graph visualization (per-cascade)
# ---------------------------------------------------------------------

def plot_cascade_graph(
    nodes_df: pd.DataFrame,
    submission_id: str,
    early_authors: Optional[List[str]] = None,
    max_nodes: int = 250,
    output_dir: str = "figures",
    filename_prefix: str = "cascade_graph",
):
    """
    Visualize a single cascade's comment tree as a graph.
    - Red node = submission (root)
    - Orange nodes = early authors (if provided)
    - Blue nodes = everyone else
    """
    _ensure_dir(output_dir)
    early_authors = set(early_authors or [])

    # Filter nodes for this cascade, then trim to max_nodes by time
    sub = nodes_df[nodes_df["submission_id"] == submission_id].sort_values("created_utc")
    root = sub[sub["is_submission"]]
    comments = sub[~sub["is_submission"]].head(max_nodes)
    trimmed = pd.concat([root, comments], ignore_index=True)

    # Build DiGraph
    G = nx.DiGraph()
    for _, row in trimmed.iterrows():
        G.add_node(
            row["node_id"],
            author=row["author"],
            is_submission=bool(row["is_submission"]),
        )

    for _, row in trimmed.iterrows():
        if pd.notna(row["parent_id"]):
            if row["parent_id"] in G.nodes:
                G.add_edge(row["parent_id"], row["node_id"])

    # Layout
    try:
        # Requires pygraphviz + graphviz installed
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Color + size mapping
    node_colors = []
    node_sizes = []
    for n, data in G.nodes(data=True):
        if data["is_submission"]:
            node_colors.append("red")
            node_sizes.append(400)
        elif data.get("author") in early_authors:
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
    plt.title(f"Cascade Graph for {submission_id}")
    plt.axis("off")

    fname = f"{filename_prefix}_{submission_id}.png"
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Cascade graph for {submission_id} saved to {out_path}")


# ---------------------------------------------------------------------
# 4. Feature vs label distributions (community-focused)
# ---------------------------------------------------------------------

def plot_feature_by_label(
    labeled_df: pd.DataFrame,
    feature_name: str,
    output_dir: str = "figures",
    filename_prefix: str = "feature_by_label",
):
    """
    Plot distribution of a single feature split by label (0 vs 1).
    Also logs a Mann-Whitney U test p-value for the difference.

    Useful for:
      - comm_concentration
      - comm_entropy
      - comm_count
    """

    _ensure_dir(output_dir)

    if feature_name not in labeled_df.columns:
        logger.warning(f"Feature '{feature_name}' not found in labeled_df.")
        return

    df = labeled_df[[feature_name, "label"]].dropna()

    if df["label"].nunique() < 2:
        logger.warning("Not enough classes to compare distributions.")
        return

    vals_0 = df[df["label"] == 0][feature_name].values
    vals_1 = df[df["label"] == 1][feature_name].values

    # Mann-Whitney U (non-parametric, robust to non-normal distributions)
    try:
        stat, p_val = mannwhitneyu(vals_0, vals_1, alternative="two-sided")
        logger.info(
            f"[{feature_name}] Mann-Whitney U test: U={stat:.3f}, p={p_val:.3e} "
            "(0=micro-viral, 1=viral)"
        )
    except Exception as e:
        logger.warning(f"Could not compute Mann-Whitney U for {feature_name}: {e}")
        p_val = None  # unused, but kept for clarity

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [vals_0, vals_1],
        labels=["micro-viral (0)", "viral (1)"],
        showfliers=True,
    )
    plt.ylabel(feature_name)
    plt.title(f"{feature_name} by label")

    fname = f"{filename_prefix}_{feature_name}.png"
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    logger.info(f"Feature-by-label plot for '{feature_name}' saved to {out_path}")
