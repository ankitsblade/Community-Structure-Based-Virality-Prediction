# run_pipeline.py

"""
End-to-end micro-virality pipeline:
- Fetch Reddit posts with PRAW
- Build cascades
- Build global user graph & communities
- Compute features
- Label cascades
- Train baseline ML models
"""

from config import DEFAULT_SUBREDDIT, POST_LIMIT
from reddit_client import fetch_submissions
from cascades import build_cascades_for_submissions
from graphs import build_global_user_graph, detect_communities, attach_communities_to_nodes
from features import build_feature_table, add_labels
from models_ml import train_baseline_models
from logger import logger


def main():
    subreddit = DEFAULT_SUBREDDIT
    logger.info(f"Fetching submissions from r/{subreddit} (limit={POST_LIMIT})…")
    submissions = fetch_submissions(subreddit, limit=POST_LIMIT, mode="top", time_filter="week")
    logger.info(f"Fetched {len(submissions)} posts.\n")

    logger.info("Building cascades…")
    nodes_df = build_cascades_for_submissions(submissions)
    logger.info("Total nodes (posts + comments):", len(nodes_df), "\n")

    logger.info("Building global user graph…")
    user_graph = build_global_user_graph(nodes_df)

    logger.info("Detecting communities…")
    partition = detect_communities(user_graph)

    logger.info("Attaching communities to nodes…")
    nodes_with_comm = attach_communities_to_nodes(nodes_df, partition)

    logger.info("Computing features…")
    feature_df = build_feature_table(nodes_with_comm)
    logger.info("Feature table shape:", feature_df.shape)

    logger.info("Creating labels…")
    labeled_df = add_labels(feature_df)
    logger.info("Labeled cascades:", labeled_df["label"].value_counts(), "\n")

    logger.info("Training baseline ML models…")
    train_baseline_models(labeled_df)

    logger.info("\nDone ✅")


if __name__ == "__main__":
    main()
