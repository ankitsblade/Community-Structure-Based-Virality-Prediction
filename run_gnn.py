# run_gnn.py

"""
GNN training & analysis pipeline:

- Load cached cascades from data/nodes_multi.parquet
- Build global user graph & detect communities
- Compute cascade + community features
- Label cascades as micro-viral / viral
- Run:
    - Global GNN (all subreddits combined)
    - Per-subreddit GNN
    - Cross-subreddit generalization GNN
    - Feature ablation GNN runs
"""

from logger import logger
from data_io import load_nodes_df
from graphs import (
    build_global_user_graph,
    detect_communities,
    attach_communities_to_nodes,
)
from features import build_feature_table, add_labels
from gnn import (
    train_gnn_global,
    train_gnn_per_subreddit,
    train_gnn_cross_subreddit,
    train_gnn_ablation,
)


def main():
    logger.info("=== GNN PIPELINE START ===")

    # ----------------------------------------------------
    # 1. Load nodes + build communities
    # ----------------------------------------------------
    logger.info("Loading cached cascades from data/nodes_multi.parquet…")
    nodes_df = load_nodes_df("data/nodes_multi.parquet")
    logger.info(f"Loaded {len(nodes_df)} nodes.")

    if "subreddit" in nodes_df.columns:
        logger.info(f"Subreddits present: {nodes_df['subreddit'].value_counts().to_dict()}")

    logger.info("Building user graph and detecting communities…")
    G = build_global_user_graph(nodes_df)
    partition = detect_communities(G)
    nodes_with_comm = attach_communities_to_nodes(nodes_df, partition)

    # ----------------------------------------------------
    # 2. Build features + labels
    # ----------------------------------------------------
    logger.info("Building feature table…")
    feature_df = build_feature_table(nodes_with_comm)
    logger.info(f"Feature table shape: {feature_df.shape}")

    logger.info("Labeling cascades (global quantiles)…")
    labeled_df = add_labels(feature_df)
    logger.info(f"Labeled cascades: {len(labeled_df)}")
    logger.info(f"Label distribution:\n{labeled_df['label'].value_counts()}")

    # Common GNN hyperparams (tweak if needed)
    gnn_kwargs = dict(
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        batch_size=32,
        lr=1e-3,
        max_epochs=40,
    )

    # ----------------------------------------------------
    # 3. Global GNN (all subreddits combined)
    # ----------------------------------------------------
    logger.info("\n=== [GNN] GLOBAL MODEL (feature_config='full') ===")
    global_metrics = train_gnn_global(
        nodes_with_comm,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[RESULT] GLOBAL GNN metrics: {global_metrics}")

    # ----------------------------------------------------
    # 4. Per-subreddit GNN models
    # ----------------------------------------------------
    logger.info("\n=== [GNN] PER-SUBREDDIT MODELS (feature_config='full') ===")
    per_sub_metrics = train_gnn_per_subreddit(
        nodes_with_comm,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[RESULT] PER-SUBREDDIT GNN metrics: {per_sub_metrics}")

    # ----------------------------------------------------
    # 5. Cross-subreddit generalization
    # ----------------------------------------------------
    logger.info("\n=== [GNN] CROSS-SUBREDDIT GENERALIZATION (feature_config='full') ===")
    xsub_metrics = train_gnn_cross_subreddit(
        nodes_with_comm,
        labeled_df,
        feature_config="full",
        **gnn_kwargs,
    )
    logger.info(f"[RESULT] CROSS-SUB GNN metrics: {xsub_metrics}")

    # ----------------------------------------------------
    # 6. Feature ablations
    # ----------------------------------------------------
    logger.info("\n=== [GNN] FEATURE ABLATION STUDY ===")
    ablation_metrics = train_gnn_ablation(
        nodes_with_comm,
        labeled_df,
        feature_configs=["full", "structure_only", "structure_time"],
        **gnn_kwargs,
    )
    logger.info(f"[RESULT] ABLATION GNN metrics: {ablation_metrics}")

    logger.info("=== GNN PIPELINE COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
