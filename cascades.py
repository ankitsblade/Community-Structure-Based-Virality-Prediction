# microviral/cascades.py

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from collections import deque

import pandas as pd

from logger import logger

@dataclass
class CascadeNode:
    node_id: str
    parent_id: Optional[str]   # None for submission
    submission_id: str
    depth: int
    author: Optional[str]
    created_utc: float
    score: int
    is_submission: bool


def build_cascade_for_submission(submission) -> List[CascadeNode]:
    """Return a list of CascadeNode for a single submission (post + comments)."""
    submission_id = f"t3_{submission.id}"
    nodes: List[CascadeNode] = []

    # root = submission
    nodes.append(
        CascadeNode(
            node_id=submission_id,
            parent_id=None,
            submission_id=submission_id,
            depth=0,
            author=str(submission.author) if submission.author else None,
            created_utc=float(submission.created_utc),
            score=submission.score,
            is_submission=True,
        )
    )

    # full comment tree
    submission.comments.replace_more(limit=None)
    q = deque(submission.comments)

    while q:
        c = q.popleft()
        nodes.append(
            CascadeNode(
                node_id=f"t1_{c.id}",
                parent_id=c.parent_id,
                submission_id=submission_id,
                depth=c.depth,
                author=str(c.author) if c.author else None,
                created_utc=float(c.created_utc),
                score=c.score,
                is_submission=False,
            )
        )
        for r in c.replies:
            q.append(r)

    return nodes


def build_cascades_for_submissions(submissions) -> pd.DataFrame:
    """Build cascades for a list of PRAW submissions and return a single nodes DataFrame."""
    all_nodes: List[Dict[str, Any]] = []
    for i, submission in enumerate(submissions, start=1):
        logger.info(f"{i}. {submission.title[:80]}  (comments={submission.num_comments})")
        try:
            nodes = build_cascade_for_submission(submission)
            all_nodes.extend(asdict(n) for n in nodes)
        except Exception as e:
            logger.info("   Skipping due to error:", e)

    nodes_df = pd.DataFrame(all_nodes)
    return nodes_df
