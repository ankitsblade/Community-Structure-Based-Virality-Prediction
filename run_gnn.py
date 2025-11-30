# run_gnn.py

"""
GNN training & analysis pipeline (Reddit + Bluesky unified).

For Reddit:
    - Loads data/nodes_multi.parquet
    - Uses existing subreddit labels

For Bluesky:
    - Loads data/bluesky/nodes_multi.parquet
    - Treats primary hashtag as 'subreddit'
    - Runs identical GNN pipeline

Pipeline:
- Load nodes
- Build global user graph
- Detect communities
- Attach communities to nodes
- Build features + labels (virality / micro-virality)
- Run:
    - Global GNN
    - Per-subreddit (or per-hashtag) GNN
    - Cross-subreddit generalization GNN
    - Ablation study
"""

import argparse
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
from gnn import (
    train_gnn_global,
    train_gnn_per_subreddit,
    train_gnn_cross_subreddit,
    train_gnn_ablation,
)


# --------------------------------------------------------
# Argument parser
# --------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNN virality pipeline.")

    parser.add_argument(
        "--source",
        choices=["reddit", "bluesky"],
        default="reddit",
        help="Choose data source pipeline: reddit or bluesky",
    )

    parser.add_argument(
        "--nodes-path",
        type=str,
        default=None,
        help="Override path to nodes_multi.parquet (optional)",
    )

    # You can add more hyperparameters here (epochs, lr, etc.)

    return parser.parse_args()


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=== GNN PIPELINE START ===")
    logger.info(f"Selected source: {args.source.upper()}")

    # ----------------------------------------------------
    # 1. Load nodes
    # ----------------------------------------------------
    if args.nodes_path:
        nodes_path = args.nodes_path
    else:
        if args.source == "reddit":
            nodes_path = "data/nodes_multi.parquet"
        else:
            nodes_path = "data/bluesky/nodes_multi.parquet"

    logger.info(f"Loading cascades from {nodes_path}…")
    nodes_df = load_nodes_df(nodes_path)
    logger.info(f"Loaded {len(nodes_df)} node events.")

    # Map Bluesky hashtag → subreddit
    if args.source == "bluesky":
        if "subreddit" not in nodes_df.columns:
            if "primary_hashtag" in nodes_df.columns:
                logger.info("Mapping primary_hashtag → subreddit for Bluesky pipeline.")
                nodes_df["subreddit"] = nodes_df["primary_hashtag"]
            else:
                raise ValueError("Bluesky nodes file missing both 'subreddit' and 'primary_hashtag' columns.")

    if "subreddit" in nodes_df.columns:
        logger.info(f"Communities present: {nodes_df['subreddit'].value_counts().to_dict()}")

    # ----------------------------------------------------
    # 2. Build user graph + detect structural communities
    # ----------------------------------------------------
    logger.info("Building global user graph…")
    G = build_global_user_graph(nodes_df)

    logger.info("Detecting structural communities (Louvain)…")
    partition = detect_communities(G)

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

    # ----------------------------------------------------
    # 5. Per-subreddit (per-hashtag) GNNs
    # ----------------------------------------------------
    logger.info("\n=== PER-SUBREDDIT / PER-HASHTAG GNN TRAINING ===")
    per_sub_metrics = train_gnn_per_subreddit(
        nodes_df,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[PER-SUBREDDIT] Metrics: {per_sub_metrics}")

    # ----------------------------------------------------
    # 6. Cross-subreddit (cross-hashtag) generalization
    # ----------------------------------------------------
    logger.info("\n=== CROSS-SUBREDDIT / CROSS-HASHTAG GENERALIZATION ===")
    cross_metrics = train_gnn_cross_subreddit(
        nodes_df,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[CROSS-SUBREDDIT] Metrics: {cross_metrics}")

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

    logger.info("=== GNN PIPELINE COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
