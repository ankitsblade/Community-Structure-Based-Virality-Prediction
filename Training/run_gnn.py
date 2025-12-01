# run_gnn.py

"""
GNN training & analysis pipeline (Reddit only).

Pipeline:
- Load Reddit nodes from data/nodes_multi.parquet (or --nodes-path)
- Build global user graph
- Detect communities (Louvain)
- Attach communities to nodes
- Build cascade + community features
- Assign virality labels
- Compute user-graph modularity
- Run:
    - Global GNN
    - Per-subreddit GNN
    - Cross-subreddit generalization GNN
    - Ablation study
- Log GNN training to Weights & Biases (if installed and not disabled)
- Produce visuals in figs/gnn
- Dump all GNN histories + summary metrics to CSV in results/gnn
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from networkx.algorithms.community import quality as nx_quality

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
from models.gnn import (
    train_gnn_global,
    train_gnn_per_subreddit,
    train_gnn_cross_subreddit,
    train_gnn_ablation,
)
from viz.visuals import (
    plot_cascade_size_distribution,
    plot_depth_distribution,
    plot_subreddit_counts,
    plot_modularity_distribution,
    plot_gnn_training,
    plot_gnn_training_auc,
    plot_gnn_metrics,
    plot_per_subreddit_gnn_metrics,
    plot_gnn_ablation_results,
)


# --------------------------------------------------------
# Argument parser
# --------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNN virality pipeline (Reddit).")

    parser.add_argument(
        "--nodes-path",
        type=str,
        default=None,
        help="Override path to Reddit nodes_multi.parquet (optional). "
             "Defaults to data/nodes_multi.parquet",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even if installed.",
    )

    return parser.parse_args()


# --------------------------------------------------------
# CSV dump helpers
# --------------------------------------------------------

def dump_gnn_histories_to_csv(
    global_metrics: dict,
    per_sub_metrics: dict,
    cross_metrics: dict,
    ablation_metrics: dict,
    out_path: Path,
    source: str = "reddit",
):
    """
    Dump per-epoch histories for all GNN experiments into a single CSV.

    Columns:
        source, category, name, epoch, train_loss, val_loss, train_auc, val_auc
    """
    rows = []

    def _append_history(category: str, name: str, metrics: dict):
        history = metrics.get("history", {})
        epochs = history.get("epoch", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_auc = history.get("train_auc", [])
        val_auc = history.get("val_auc", [])

        for i in range(len(epochs)):
            rows.append(
                {
                    "source": source,
                    "category": category,
                    "name": name,
                    "epoch": epochs[i],
                    "train_loss": train_loss[i] if i < len(train_loss) else None,
                    "val_loss": val_loss[i] if i < len(val_loss) else None,
                    "train_auc": train_auc[i] if i < len(train_auc) else None,
                    "val_auc": val_auc[i] if i < len(val_auc) else None,
                }
            )

    # Global
    if global_metrics:
        _append_history("global", "global_full", global_metrics)

    # Per-subreddit
    for sub, m in per_sub_metrics.items():
        _append_history("per_subreddit", sub, m)

    # Cross-subreddit (held-out)
    for heldout, m in cross_metrics.items():
        name = f"heldout_{heldout}"
        _append_history("cross_subreddit", name, m)

    # Ablation (feature configs)
    for cfg, m in ablation_metrics.items():
        _append_history("ablation", cfg, m)

    if not rows:
        logger.warning(f"No GNN histories to dump to CSV at {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Dumped GNN histories to {out_path}")


def dump_gnn_summary_to_csv(
    global_metrics: dict,
    per_sub_metrics: dict,
    cross_metrics: dict,
    ablation_metrics: dict,
    out_path: Path,
    source: str = "reddit",
):
    """
    Dump final train/val/test AUCs for all experiments into a CSV.

    Columns:
        source, category, name, train_auc, val_auc, test_auc
    """
    rows = []

    def _extract_row(category: str, name: str, metrics: dict):
        if not metrics:
            return
        rows.append(
            {
                "source": source,
                "category": category,
                "name": name,
                "train_auc": metrics.get("train_auc"),
                "val_auc": metrics.get("val_auc"),
                "test_auc": metrics.get("test_auc"),
            }
        )

    # Global
    _extract_row("global", "global_full", global_metrics)

    # Per-subreddit
    for sub, m in per_sub_metrics.items():
        _extract_row("per_subreddit", sub, m)

    # Cross-subreddit
    for heldout, m in cross_metrics.items():
        name = f"heldout_{heldout}"
        _extract_row("cross_subreddit", name, m)

    # Ablation
    for cfg, m in ablation_metrics.items():
        _extract_row("ablation", cfg, m)

    if not rows:
        logger.warning(f"No GNN summary metrics to dump to CSV at {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Dumped GNN summary metrics to {out_path}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=== GNN PIPELINE START (REDDIT) ===")

    # ----------------------------------------------------
    # 1. Load nodes (Reddit only)
    # ----------------------------------------------------
    if args.nodes_path:
        nodes_path = args.nodes_path
    else:
        nodes_path = "data/nodes_multi.parquet"

    logger.info(f"Loading cascades from {nodes_path}…")
    nodes_df = load_nodes_df(nodes_path)
    logger.info(f"Loaded {len(nodes_df)} node events.")

    if "subreddit" in nodes_df.columns:
        logger.info(
            f"Subreddit distribution: "
            f"{nodes_df['subreddit'].value_counts().to_dict()}"
        )

    figs_dir = "figs/gnn"

    # ----------------------------------------------------
    # Basic cascade / subreddit distributions
    # ----------------------------------------------------
    try:
        plot_cascade_size_distribution(nodes_df, out_dir=figs_dir)
        plot_depth_distribution(nodes_df, out_dir=figs_dir)
        plot_subreddit_counts(nodes_df, out_dir=figs_dir)
        logger.info(f"Saved basic cascade/subreddit distribution plots to {figs_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate basic distribution plots: {e}")

    # ----------------------------------------------------
    # 2. Build user graph + detect structural communities
    # ----------------------------------------------------
    logger.info("Building global user graph…")
    G = build_global_user_graph(nodes_df)

    logger.info("Detecting structural communities (Louvain)…")
    partition = detect_communities(G)

    # Compute global modularity of user graph
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

    logger.info("Attaching communities to nodes…")
    nodes_df = attach_communities_to_nodes(nodes_df, partition)

    # ----------------------------------------------------
    # 3. Build features + virality labels
    # ----------------------------------------------------
    logger.info("Building feature table…")
    feature_df = build_feature_table(nodes_df)
    logger.info(f"Feature table shape: {feature_df.shape}")

    logger.info("Assigning virality labels via global quantiles…")
    labeled_df = add_labels(feature_df)

    logger.info(f"Label counts:\n{labeled_df['label'].value_counts()}")

    # Common GNN hyperparameters
    gnn_kwargs = dict(
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        batch_size=32,
        lr=1e-3,
        max_epochs=40,
        use_wandb=not args.no_wandb,
    )

    # ----------------------------------------------------
    # 4. Global GNN
    # ----------------------------------------------------
    logger.info("\n=== GLOBAL GNN TRAINING ===")
    global_metrics = train_gnn_global(
        nodes_df,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[GLOBAL GNN] Metrics: {global_metrics}")

    # Visuals: global GNN training & metrics
    history = global_metrics.get("history", {})
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_aucs = history.get("train_auc", [])
    val_aucs = history.get("val_auc", [])

    try:
        plot_gnn_training(train_losses, val_losses, out_dir=figs_dir, tag="global")
        plot_gnn_training_auc(train_aucs, val_aucs, out_dir=figs_dir, tag="global")
        summary_metrics = {
            k: v
            for k, v in global_metrics.items()
            if k in {"train_auc", "val_auc", "test_auc"}
        }
        plot_gnn_metrics(summary_metrics, out_dir=figs_dir, tag="global")
        logger.info(f"Saved global GNN training/eval plots to {figs_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate global GNN visuals: {e}")

    # ----------------------------------------------------
    # 5. Per-subreddit GNNs
    # ----------------------------------------------------
    logger.info("\n=== PER-SUBREDDIT GNN TRAINING ===")
    per_sub_metrics = train_gnn_per_subreddit(
        nodes_df,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[PER-SUBREDDIT] Metrics: {per_sub_metrics}")

    # Visuals: per-subreddit performance
    try:
        plot_per_subreddit_gnn_metrics(per_sub_metrics, metric="test_auc", out_dir=figs_dir)
        logger.info(f"Saved per-subreddit GNN metric plots to {figs_dir}")
    except Exception as e:
        logger.warning(f"Failed to plot per-subreddit GNN metrics: {e}")

    # ----------------------------------------------------
    # 6. Cross-subreddit generalization
    # ----------------------------------------------------
    logger.info("\n=== CROSS-SUBREDDIT GENERALIZATION ===")
    cross_metrics = train_gnn_cross_subreddit(
        nodes_df,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[CROSS-SUBREDDIT] Metrics: {cross_metrics}")

    # (Optional) you can later visualize cross_metrics if you want

    # ----------------------------------------------------
    # 7. Feature ablation
    # ----------------------------------------------------
    logger.info("\n=== FEATURE ABLATION STUDY ===")
    ablation_metrics = train_gnn_ablation(
        nodes_df,
        labeled_df,
        feature_configs=["full", "structure_only", "structure_time"],
        **gnn_kwargs,
    )
    logger.info(f"[ABLATION] Metrics: {ablation_metrics}")

    # Visuals: ablation performance
    try:
        plot_gnn_ablation_results(ablation_metrics, metric="test_auc", out_dir=figs_dir)
        logger.info(f"Saved GNN ablation plots to {figs_dir}")
    except Exception as e:
        logger.warning(f"Failed to plot GNN ablation results: {e}")

    # ----------------------------------------------------
    # 8. Dump CSVs (histories + summaries)
    # ----------------------------------------------------
    results_dir = Path("results/gnn")
    histories_csv = results_dir / "gnn_histories.csv"
    summary_csv = results_dir / "gnn_summary.csv"

    dump_gnn_histories_to_csv(
        global_metrics,
        per_sub_metrics,
        cross_metrics,
        ablation_metrics,
        out_path=histories_csv,
        source="reddit",
    )

    dump_gnn_summary_to_csv(
        global_metrics,
        per_sub_metrics,
        cross_metrics,
        ablation_metrics,
        out_path=summary_csv,
        source="reddit",
    )

    logger.info("=== GNN PIPELINE COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
