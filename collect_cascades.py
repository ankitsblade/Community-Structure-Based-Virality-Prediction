# collect_cascades.py

"""
Multi-subreddit data collection script.

- For each subreddit in config.SUBREDDITS:
    - fetch submissions with PRAW
    - build cascades (nodes_df) with a 'subreddit' column
- Concatenate all subreddits into a single DataFrame
- Save to disk as data/nodes_multi.parquet

Run this occasionally to refresh your dataset.
"""

import pandas as pd

from logger import logger
from config import SUBREDDITS, POSTS_PER_SUBREDDIT
from reddit_client import fetch_submissions
from cascades import build_cascades_for_submissions
from data_io import save_nodes_df


def main():
    all_nodes_dfs = []

    for subreddit in SUBREDDITS:
        logger.info(
            f"[COLLECT] Fetching submissions from r/{subreddit} "
            f"(limit={POSTS_PER_SUBREDDIT})…"
        )
        submissions = fetch_submissions(
            subreddit_name=subreddit,
            limit=POSTS_PER_SUBREDDIT,
            mode="top",
            time_filter="week",
        )
        logger.info(f"[COLLECT] Fetched {len(submissions)} posts from r/{subreddit}.")

        logger.info(f"[COLLECT] Building cascades for r/{subreddit}…")
        nodes_df_sub = build_cascades_for_submissions(submissions, subreddit_name=subreddit)
        logger.info(f"[COLLECT] r/{subreddit}: extracted {len(nodes_df_sub)} nodes.")
        all_nodes_dfs.append(nodes_df_sub)

    if not all_nodes_dfs:
        logger.error("[COLLECT] No cascades collected from any subreddit.")
        return

    combined_nodes_df = pd.concat(all_nodes_dfs, ignore_index=True)
    logger.info(
        f"[COLLECT] Combined nodes_df: {len(combined_nodes_df)} rows across "
        f"{combined_nodes_df['subreddit'].nunique()} subreddits."
    )

    logger.info("[COLLECT] Saving combined cascades to disk…")
    save_nodes_df(combined_nodes_df, path="data/nodes_multi.parquet")

    logger.info("[COLLECT] Done ✅")


if __name__ == "__main__":
    main()
