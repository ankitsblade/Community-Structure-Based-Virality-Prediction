# collect_cascades.py

"""
Multi-subreddit data collection script for microviral.

For each subreddit in config.SUBREDDITS:
    - fetch up to POSTS_PER_SUBREDDIT submissions with PRAW
    - build cascades (nodes_df) with a 'subreddit' column
All subreddits are concatenated into a single DataFrame and saved as:

    data/nodes_multi.parquet

This file is then used by run_pipeline.py and run_gnn.py.

Usage:
    uv run collect_cascades.py
"""

from typing import List

import pandas as pd

from config import SUBREDDITS, POSTS_PER_SUBREDDIT
from reddit_client import fetch_submissions
from cascades import build_cascades_for_submissions
from data_io import save_nodes_df
from logger import logger


def collect_for_subreddit(subreddit: str) -> pd.DataFrame:
    """
    Collect cascades for a single subreddit and return a nodes_df
    with at least the following columns:

        - submission_id
        - node_id / parent_id
        - depth, author, created_utc, score, etc.
        - subreddit (string)

    If nothing is collected, returns an empty DataFrame.
    """
    logger.info(
        f"[COLLECT] Fetching submissions from r/{subreddit} "
        f"(limit={POSTS_PER_SUBREDDIT})…"
    )

    submissions = fetch_submissions(
        subreddit_name=subreddit,
        limit=POSTS_PER_SUBREDDIT,
        mode="top",          # "top", "hot", or "new" depending on your design
        time_filter="year",  # widen if you want older posts; "all" is also ok
    )

    if not submissions:
        logger.warning(f"[COLLECT] No submissions returned for r/{subreddit}.")
        return pd.DataFrame()

    logger.info(f"[COLLECT] Building cascades for r/{subreddit}…")
    nodes_df = build_cascades_for_submissions(
        submissions,
        subreddit_name=subreddit,
    )

    if nodes_df is None or nodes_df.empty:
        logger.warning(f"[COLLECT] No cascades built for r/{subreddit}.")
        return pd.DataFrame()

    logger.info(
        f"[COLLECT] r/{subreddit}: built "
        f"{nodes_df['submission_id'].nunique()} cascades "
        f"({len(nodes_df)} nodes)."
    )
    return nodes_df


def main() -> None:
    all_nodes: List[pd.DataFrame] = []

    logger.info(
        f"[COLLECT] Starting multi-subreddit collection for "
        f"{len(SUBREDDITS)} subreddits."
    )

    for subreddit in SUBREDDITS:
        try:
            df_sub = collect_for_subreddit(subreddit)
        except Exception as e:
            logger.warning(f"[COLLECT] Skipping r/{subreddit} due to error: {e}")
            continue

        if df_sub is not None and not df_sub.empty:
            all_nodes.append(df_sub)
        else:
            logger.warning(f"[COLLECT] r/{subreddit} produced empty DataFrame.")

    if not all_nodes:
        raise RuntimeError(
            "[COLLECT] No cascades collected from any subreddit – "
            "check Reddit credentials / SUBREDDITS / network."
        )

    combined_nodes_df = pd.concat(all_nodes, ignore_index=True)

    logger.info(
        f"[COLLECT] Combined nodes_df: {len(combined_nodes_df)} rows across "
        f"{combined_nodes_df['submission_id'].nunique()} cascades and "
        f"{combined_nodes_df['subreddit'].nunique()} subreddits."
    )

    logger.info("[COLLECT] Saving combined cascades to disk…")
    save_nodes_df(combined_nodes_df, path="data/nodes_multi.parquet")

    logger.info("[COLLECT] Done ✅")


if __name__ == "__main__":
    main()
