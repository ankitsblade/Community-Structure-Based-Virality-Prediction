# run_pipeline.py

"""
End-to-end analysis pipeline (no Reddit calls):

- Load cached cascades from data/nodes_multi.parquet
- Build global user graph & detect communities
- Compute cascade + community features
- Label cascades as micro-viral / viral
- Train baseline ML models (community + dynamics)
- Train community-only models
- Generate figures for distributions & feature importance
- Optionally plot one example cascade graph
"""

from logger import logger
from config import EARLY_K
from data_io import load_nodes_df
from graphs import (
    build_global_user_graph,
    detect_communities,
    attach_communities_to_nodes,
)
from features import (
    build_feature_table,
    add_labels,
    get_early_adopters,
)
from models_ml import (
    train_baseline_models,
    train_community_only_model,
    inspect_logreg_coeffs_community,
)
from visuals import (
    plot_cascade_size_distribution,
    plot_label_balance,
    plot_feature_importances,
    plot_cascade_graph,
    plot_feature_by_label,
)


def main():
    logger.info("Loading cached multi-subreddit cascades from data/nodes_multi.parquet…")
    nodes_df = load_nodes_df("data/nodes_multi.parquet")
    logger.info(f"Total nodes loaded: {len(nodes_df)}")
    if "subreddit" in nodes_df.columns:
        logger.info(f"Subreddits present: {nodes_df['subreddit'].value_counts().to_dict()}")

    logger.info("Building user graph and communities across all subreddits…")
    user_graph = build_global_user_graph(nodes_df)
    partition = detect_communities(user_graph)

    logger.info("Attaching community IDs to nodes…")
    nodes_with_comm = attach_communities_to_nodes(nodes_df, partition)

    logger.info("Computing feature table…")
    feature_df = build_feature_table(nodes_with_comm)
    logger.info(f"Feature table shape: {feature_df.shape}")

    plot_cascade_size_distribution(feature_df)

    logger.info("Labeling cascades (global quantiles)…")
    labeled_df = add_labels(feature_df)
    logger.info(f"Label distribution (all subs):\n{labeled_df['label'].value_counts()}")

    plot_label_balance(labeled_df)

    logger.info("Running baseline ML models (community + dynamics)…")
    auc_logreg, auc_rf, rf_model, feature_cols = train_baseline_models(labeled_df)
    plot_feature_importances(feature_cols, rf_model.feature_importances_)

    logger.info("Running community-only models for interpretability…")
    train_community_only_model(labeled_df)
    inspect_logreg_coeffs_community(labeled_df)

    # Community feature distributions by label
    for feat in ["comm_concentration", "comm_entropy", "comm_count"]:
        plot_feature_by_label(labeled_df, feat)

    # Optional: visualise one example cascade graph
    if not labeled_df.empty:
        example_sub = labeled_df["submission_id"].iloc[0]
        early = get_early_adopters(nodes_with_comm, example_sub, k=EARLY_K)
        plot_cascade_graph(
            nodes_with_comm,
            submission_id=example_sub,
            early_authors=early,
            max_nodes=250,
        )

    logger.info("Multi-subreddit analysis pipeline complete. ✅")


if __name__ == "__main__":
    main()
