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

SUBREDDITS = [
    "AskReddit", "AskScience", "AskHistorians", "AskEconomics", "explainlikeimfive",
    "todayilearned", "worldnews", "news", "technology", "science",
    "space", "Futurology", "changemyview", "LifeProTips", "philosophy",
    "dataisbeautiful", "interestingasfuck", "mildlyinteresting", "nextfuckinglevel", "awesome",
    "movies", "television", "Documentaries", "Film", "TrueFilm",
    "gaming", "pcgaming", "buildapc", "PS5", "NintendoSwitch",
    "india", "indiasocial", "IndianFood", "bangalore", "hyderabad",
    "books", "WritingPrompts", "nosleep", "shortstories", "FanFiction",
    "Music", "listentothis", "hiphopheads", "ClassicalMusic", "EDM",
    "food", "Cooking", "AskCulinary", "baking", "coffee",
    "sports", "soccer", "nba", "cricket", "formula1",
    "fitness", "bodyweightfitness", "running", "cycling", "yoga",
    "personalfinance", "Economics", "stocks", "wallstreetbets", "CryptoCurrency",
    "learnprogramming", "programming", "cpp", "python", "MachineLearning",
    "gadgets", "android", "apple", "iOS", "Art",
    "drawing", "sketches", "Illustration", "Design", "DIY",
    "lifehacks", "howto", "homelab", "malelivingspace", "history",
    "historyporn", "AncientHistory", "TheWayWeWere", "relationship_advice", "relationships",
    "dating_advice", "socialskills", "decidingtobebetter", "productivity", "getdisciplined",
    "selfimprovement", "Stoicism", "Zen", "psychology", "neuro",
    "behavioralscience", "ADHD", "travel", "solotravel", "IWantToLearn",
    "IAmA", "UpliftingNews", "CyberSecurity", "netsec", "opensource",
    "Linux", "TrueReddit", "DepthHub", "Bestof", "EverythingScience",
    "TheoreticalPhysics", "wholesomememes", "memes", "funny", "comedy",
    "dankmemes", "nature", "earthporn", "gardening", "botany",
    "wildlife", "UrbanExploration", "architecture", "Houseplants", "interiordecorating",
    "LandscapePhotography", "askphilosophy", "Literature", "linguistics", "math",
    "statistics", "datascience", "bigdata", "cs50", "computervision",
    "deeplearning", "internetisbeautiful", "educationalgifs", "casualconversation", "TrueAskReddit",
]

# Hard cap per subreddit (API / design constraint)
POSTS_PER_SUBREDDIT = 500
