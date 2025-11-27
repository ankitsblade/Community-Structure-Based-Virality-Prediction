# microviral/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# Reddit API credentials
# -------------------------------------------------------

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
USER_AGENT = os.getenv("USER_AGENT", "microviral_app:v1.0 (by u/your_username)")

# -------------------------------------------------------
# Core config
# -------------------------------------------------------

# You can still use these for single-subreddit experiments if needed
DEFAULT_SUBREDDIT = "politics"
POST_LIMIT = 500

# Number of distinct early adopters to consider
EARLY_K = 10

# Optional: for visual trimming
MAX_NODES_PER_CASCADE = 250

# Ignore cascades with fewer than this many comments when labeling
MIN_COMMENTS_FOR_LABEL = 20

# Quantiles for micro-viral and viral labels
MICRO_PERC = 0.80
VIRAL_PERC = 0.95

# Random seed for splits / models
RANDOM_STATE = 42

# -------------------------------------------------------
# Multi-subreddit setup
# -------------------------------------------------------

SUBREDDITS = ["politics", "worldnews", "technology"]
POSTS_PER_SUBREDDIT = 2000  # adjust based on how big you want the dataset
