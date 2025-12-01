# collect_arctic_cascades.py
import json
import os
from collections import defaultdict, deque
from typing import Dict, Iterable, List

import pandas as pd
import zstandard as zstd

from config import SUBREDDITS, POSTS_PER_SUBREDDIT
from util.data_io import save_nodes_df
from logger import logger


def iter_zst_jsonl(path: str):
    """
    Stream .zst NDJSON files from your reddit dump.
    """
    logger.info(f"[ZST] Reading {path}")
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**20)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception as e:
                        logger.warning(f"[ZST] JSON parse error in {path}: {e}")


def list_zst_files(folder: str):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".zst")
    )


def load_submissions(submissions_dir: str):
    target = {s.lower() for s in SUBREDDITS}
    per_sub_count = defaultdict(int)
    submissions = {}

    for path in list_zst_files(submissions_dir):
        for rec in iter_zst_jsonl(path):

            sr = str(rec.get("subreddit", "")).lower()
            if sr not in target:
                continue

            if per_sub_count[sr] >= POSTS_PER_SUBREDDIT:
                continue

            rid = rec.get("id")
            if rid is None:
                continue

            submission_id = f"t3_{rid}"
            submissions[submission_id] = rec
            per_sub_count[sr] += 1

    logger.info(f"[ARCTIC] Total submissions collected: {len(submissions)}")
    return submissions


def load_comments(comments_dir: str, submissions: Dict[str, Dict]):
    valid = set(submissions.keys())
    comments_by_sub = defaultdict(list)

    for path in list_zst_files(comments_dir):
        for rec in iter_zst_jsonl(path):

            link_id = rec.get("link_id")
            if link_id not in valid:
                continue

            comments_by_sub[link_id].append(rec)

    logger.info(
        f"[ARCTIC] Loaded comments for {len(comments_by_sub)} submissions "
        f"(total comments: {sum(len(v) for v in comments_by_sub.values())})"
    )
    return comments_by_sub


def build_cascade(submission_id: str, subrec: Dict, comments: List[Dict]) -> List[Dict]:
    nodes = []

    subreddit = subrec.get("subreddit")
    author = subrec.get("author")
    created_utc = float(subrec.get("created_utc", 0.0))
    score = int(subrec.get("score", 0))

    nodes.append(dict(
        node_id=submission_id,
        parent_id=None,
        submission_id=submission_id,
        depth=0,
        author=str(author) if author else None,
        created_utc=created_utc,
        score=score,
        is_submission=True,
        subreddit=str(subreddit)
    ))

    children = defaultdict(list)
    for c in comments:
        pid = c.get("parent_id")
        if pid:
            children[pid].append(c)

    q = deque([(submission_id, 0)])
    seen = {submission_id}

    while q:
        pid, d = q.popleft()

        for c in children.get(pid, []):
            cid_raw = c.get("id")
            if not cid_raw:
                continue

            node_id = f"t1_{cid_raw}"
            if node_id in seen:
                continue
            seen.add(node_id)

            nodes.append(dict(
                node_id=node_id,
                parent_id=c.get("parent_id"),
                submission_id=submission_id,
                depth=d + 1,
                author=str(c.get("author")) if c.get("author") else None,
                created_utc=float(c.get("created_utc", created_utc)),
                score=int(c.get("score", 0)),
                is_submission=False,
                subreddit=str(subreddit)
            ))

            q.append((node_id, d + 1))

    return nodes


def build_all(submissions: Dict[str, Dict], comments_by_sub):
    all_nodes = []
    for sid, subrec in submissions.items():
        try:
            nodes = build_cascade(sid, subrec, comments_by_sub.get(sid, []))
            all_nodes.extend(nodes)
        except Exception as e:
            logger.warning(f"[ARCTIC] Skipped {sid}: {e}")

    df = pd.DataFrame(all_nodes)
    logger.info(f"[ARCTIC] Final cascade nodes: {len(df)}")
    return df


def main():
    submissions_dir = "reddit/submissions"
    comments_dir = "reddit/comments"

    logger.info("[ARCTIC] Collecting cascades from dump…")

    subs = load_submissions(submissions_dir)
    comments = load_comments(comments_dir, subs)
    df = build_all(subs, comments)

    save_nodes_df(df, "data/nodes_multi.parquet")
    logger.info("[ARCTIC] DONE — saved to data/nodes_multi.parquet")


if __name__ == "__main__":
    main()
