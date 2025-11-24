# microviral/reddit_client.py

import praw
from config import CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD, USER_AGENT

def get_reddit():
    """Create and return a configured PRAW Reddit client."""
    if not all([CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD]):
        raise RuntimeError("Missing Reddit API credentials in environment variables.")
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent=USER_AGENT,
    )
    return reddit


def fetch_submissions(subreddit_name: str, limit: int = 100, mode: str = "top", time_filter: str = "week"):
    """
    Fetch submissions from a subreddit.
    mode: 'hot', 'new', 'top'
    time_filter: 'day', 'week', 'month', 'year', 'all' (used only for 'top')
    """
    reddit = get_reddit()
    subreddit = reddit.subreddit(subreddit_name)

    if mode == "hot":
        submissions = list(subreddit.hot(limit=limit))
    elif mode == "new":
        submissions = list(subreddit.new(limit=limit))
    elif mode == "top":
        submissions = list(subreddit.top(time_filter=time_filter, limit=limit))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return submissions
