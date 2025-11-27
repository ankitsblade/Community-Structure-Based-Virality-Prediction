# run_pipeline.py

"""
End-to-end analysis pipeline (no Reddit calls):

- Load cached cascades from data/nodes_multi.parquet
- Build global user graph & detect communities
- Compute cascade + community features
- Label cascades as micro-viral / viral
- Train baseline ML models (community + dynamics)
- Train community-only models
- Generate global figures for distributions & feature importance
- Visualize one example cascade graph
- Per-subreddit analysis (metrics + logs, minimal plots)
- Grouped multi-subreddit plots (label balance + feature-by-label)
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
    plot_label_balance_multi,
    plot_feature_by_label_multi,
)


def main():
    # ----------------------------------------------------
    # 1. Load cached cascades
    # ----------------------------------------------------
    logger.info("Loading cached multi-subreddit cascades from data/nodes_multi.parquet…")
    nodes_df = load_nodes_df("data/nodes_multi.parquet")
    logger.info(f"Total nodes loaded: {len(nodes_df)}")
    if "subreddit" in nodes_df.columns:
        logger.info(f"Subreddits present: {nodes_df['subreddit'].value_counts().to_dict()}")

    # ----------------------------------------------------
    # 2. Global user graph + communities
    # ----------------------------------------------------
    logger.info("Building user graph and communities across all subreddits…")
    user_graph = build_global_user_graph(nodes_df)
    partition = detect_communities(user_graph)

    logger.info("Attaching community IDs to nodes…")
    nodes_with_comm = attach_communities_to_nodes(nodes_df, partition)

    # ----------------------------------------------------
    # 3. Features + labeling (global)
    # ----------------------------------------------------
    logger.info("Computing feature table…")
    feature_df = build_feature_table(nodes_with_comm)
    logger.info(f"Feature table shape: {feature_df.shape}")

    # Global cascade size distribution
    plot_cascade_size_distribution(feature_df)

    logger.info("Labeling cascades (global quantiles)…")
    labeled_df = add_labels(feature_df)
    logger.info(f"Number of labeled cascades: {len(labeled_df)}")
    logger.info(f"Label distribution (all subs):\n{labeled_df['label'].value_counts()}")

    # Global label balance
    plot_label_balance(labeled_df)

    # ----------------------------------------------------
    # 4. Global models (all subreddits combined)
    # ----------------------------------------------------
    logger.info("Running baseline ML models (community + dynamics) on ALL subreddits…")
    auc_logreg, auc_rf, rf_model, feature_cols = train_baseline_models(labeled_df)
    logger.info(
        f"[GLOBAL] Baseline LogReg AUC={auc_logreg:.4f}, RandomForest AUC={auc_rf:.4f}"
    )
    plot_feature_importances(feature_cols, rf_model.feature_importances_)

    logger.info("Running community-only models for interpretability (ALL subreddits)…")
    train_community_only_model(labeled_df)
    inspect_logreg_coeffs_community(labeled_df)

    # Community feature distributions by label (global)
    for feat in ["comm_concentration", "comm_entropy", "comm_count"]:
        plot_feature_by_label(labeled_df, feat)

    # ----------------------------------------------------
    # 5. Example cascade graph (global)
    # ----------------------------------------------------
    if not labeled_df.empty:
        example_sub = labeled_df["submission_id"].iloc[0]
        early = get_early_adopters(nodes_with_comm, example_sub, k=EARLY_K)
        plot_cascade_graph(
            nodes_with_comm,
            submission_id=example_sub,
            early_authors=early,
            max_nodes=250,
        )

    # ----------------------------------------------------
    # 6. Per-subreddit analysis (metrics only, minimal plots)
    # ----------------------------------------------------
    if "subreddit" not in labeled_df.columns:
        logger.warning("No 'subreddit' column in labeled_df – skipping per-subreddit analysis.")
        logger.info("Analysis pipeline complete. ✅")
        return

    logger.info("Starting per-subreddit analysis (metrics + logs, minimal figures)…")

    for sub in labeled_df["subreddit"].dropna().unique():
        sub_df = labeled_df[labeled_df["subreddit"] == sub]

        logger.info(f"\n[Per-subreddit] r/{sub}")
        logger.info(f"  Number of labeled cascades: {len(sub_df)}")
        logger.info(f"  Label distribution:\n{sub_df['label'].value_counts()}")

        # Need at least both classes
        if sub_df["label"].nunique() < 2:
            logger.warning(f"  r/{sub}: only one label present – skipping ML models.")
            continue

        if len(sub_df) < 50:
            logger.warning(
                f"  r/{sub}: only {len(sub_df)} cascades – results may be unstable; running anyway."
            )

        # Per-subreddit baseline models (community + dynamics)
        try:
            logger.info(f"  Training baseline models for r/{sub}…")
            auc_logreg_sub, auc_rf_sub, _, _ = train_baseline_models(sub_df)
            logger.info(
                f"  r/{sub}: Baseline LogReg AUC={auc_logreg_sub:.4f}, "
                f"RF AUC={auc_rf_sub:.4f}"
            )
        except ValueError as e:
            logger.warning(f"  r/{sub}: skipping baseline models due to error: {e}")

        # Per-subreddit community-only models
        try:
            logger.info(f"  Training community-only models for r/{sub}…")
            train_community_only_model(sub_df)
            inspect_logreg_coeffs_community(sub_df)
        except ValueError as e:
            logger.warning(f"  r/{sub}: skipping community-only models due to error: {e}")

    # ----------------------------------------------------
    # 7. Grouped multi-subreddit plots (compact visuals)
    # ----------------------------------------------------
    logger.info("Creating grouped multi-subreddit plots…")

    # One grouped label-balance plot
    plot_label_balance_multi(labeled_df)

    # One grouped feature-by-label plot per key community feature
    for feat in ["comm_concentration", "comm_entropy", "comm_count"]:
        plot_feature_by_label_multi(labeled_df, feat)

    logger.info("Per-subreddit analysis complete. ✅")
    logger.info("Full analysis pipeline complete. ✅")


if __name__ == "__main__":
    main()
