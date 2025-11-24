# microviral/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Reddit API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
USER_AGENT = os.getenv("USER_AGENT", "microviral_app:v1.0 (by u/your_username)")

# Data / model config
DEFAULT_SUBREDDIT = "politics"
POST_LIMIT = 500                # number of submissions to pull
EARLY_K = 10                    # number of distinct early adopters
MAX_NODES_PER_CASCADE = 250     # for visualisation / sanity checks
MIN_COMMENTS_FOR_LABEL = 20     # ignore very tiny cascades
MICRO_PERC = 0.80               # 80th percentile
VIRAL_PERC = 0.95               # 95th percentile
RANDOM_STATE = 42
