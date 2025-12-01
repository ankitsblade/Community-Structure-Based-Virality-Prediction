# 🧵 Community-Structure-Based Virality Prediction

This project predicts whether a Reddit post becomes **micro-viral** or **viral** using:

* Multi-subreddit cascade extraction
* Community structure analysis via user-graph Louvain detection
* Cascade structural + temporal features
* Classical ML models
* Graph Neural Networks (GraphSAGE)
* Cross-subreddit and early-cascade prediction

The pipeline supports **two data sources**:

1. **Arctic Shift Reddit Comment/Submission Dump** (research dump)
2. **Live PRAW API collection** (for fresh cascades)

---

# 📁 Final Project Structure

```
Community-Structure-Based-Virality-Prediction/
│
├── collect_cascades.py              # Reddit data collection using PRAW
├── data_util/
│   ├── __init__.py
│   ├── cascades.py                  # Submission → cascade extraction
│   ├── collect_cascades_pushshift.py
│   ├── collect_cascades.py          # Multi-subreddit cascade loader
│   └── reddit_client.py             # PRAW client + API helpers
│
├── dataset/                         # Arctic Shift Dataset (2023-11 reddit data)
│
├── features/
│   ├── __init__.py
│   └── features.py                  # Structural, temporal, community features
│
├── figures/
│   ├── classicml/                   # Classical ML visuals
│   ├── gnn/                         # GNN visuals
│   ├── figures_old/
│   └── reddit/
│
├── misc/
│   └── graphviz.sh                  # To Download graphviz on non-sudo compute
│
├── models/
│   ├── __init__.py
│   ├── gnn.py                       # GraphSAGE + GNN dataset builder
│   └── models_ml.py                 # Logistic Regression + Random Forest
│
├── notebooks/
│   └── Experiments.ipynb            # Scratch experiments
│
├── Training/
│   ├── run_gnn.py                   # GNN pipeline runner
│   └── run_pipeline.py              # Classical ML pipeline runner
│
├── util/
│   ├── __init__.py
│   ├── data_io.py                   # Save/load utilities
│   ├── graphs.py                    # User graph + Louvain communities
│   └── metrics.py                   # ROC-AUC helper
│
├── viz/
│   └── visuals.py                   # All visualizations
│
│── config.py                        # Subreddit list + constants
│── logger.py                        # Logging setup
│── microviral.log                   # Log file
│── README.md
│── pyproject.toml
│── .env
│── .gitignore
└── uv.lock
```

---

# 🧊 Data Sources Used

### **1️⃣ Arctic Shift Reddit Dump**

This project uses the **Arctic Shift Reddit dataset**, a research-focused archived dump containing:

* Reddit submissions
* Reddit comments
* Author IDs
* Subreddit metadata
* Timestamps

It provides the **core comment trees** for building cascade structures.

### **2️⃣ PRAW API (Python Reddit API Wrapper)**

To supplement or refresh data, PRAW is used for:

* Fetching recent submissions
* Getting full comment trees
* Extracting reply chains
* Collecting subreddit-specific cascades

The PRAW client is defined in:

```
data_util/reddit_client.py
```

You may run only Arctic Shift data, only PRAW data, or combine both.

---

# 🚀 How to Run

This section explains how to reproduce your dataset, extract features, train models, and generate all figures.

---

## 1️⃣ Install environment using **uv**

```bash
uv sync
```

This:

* Creates `.venv/`
* Installs all dependencies from `pyproject.toml`
* Ensures consistent environment

No manual `pip install` required.

---

## 2️⃣ Add Reddit API Credentials (PRAW Use Only)

Create `.env`:

```
CLIENT_ID=xxxx
CLIENT_SECRET=xxxx
USERNAME=xxxx
PASSWORD=xxxx
USER_AGENT=microviral:v1.0
```

If using only the Arctic Shift dump, this step is optional.

---

## 3️⃣ Obtain or Generate the Dataset

### Option A — Use Arctic Shift Reddit Dump

Place the processed/parquet dump inside:

```
dataset/
```

The pipeline automatically loads it.

### Option B — Collect Fresh Cascades Using PRAW

```bash
python collect_cascades.py
```

This produces:

```
dataset/nodes_multi.parquet
```

Containing:

* All nodes (submissions + comments)
* Parent/child edges
* Author IDs
* Subreddit labels
* Timestamps, scores, depths

---

## 4️⃣ Run the Classical ML Pipeline

```bash
python Training/run_pipeline.py
```

This performs:

* User graph construction
* Louvain community detection
* Modularity computation
* Feature extraction
* Virality labeling
* Training LogReg + RandomForest
* Per-subreddit metrics
* Exhaustive visualizations

Results saved to:

```
figures/classicml/
dataset/
```

---

## 5️⃣ Run the GNN Pipeline

```bash
python Training/run_gnn.py
```

This runs:

* GraphSAGE training
* Per-subreddit GNN models
* Cross-subreddit generalization
* Feature ablation studies

Outputs saved to:

```
figures/gnn/
results/gnn/
```

---

# 🧩 Summary of the Pipeline

This project performs:

1. Multi-subreddit cascade extraction (Arctic Shift + PRAW)
2. User-graph construction (reply network)
3. Louvain community structure detection
4. Feature engineering:

   * Depth, branching, entropy
   * Temporal bursts (time to 5, time-normalized flow)
   * Community attributes
5. Global virality labeling
6. Classical ML baselines
7. Graph Neural Network modeling (GraphSAGE)
8. Cross-subreddit + per-subreddit experiments
9. Feature ablations
10. Automatic figure generation for posters/papers

---

## Poster [Submitted for CS4222 Social Computing]

![alt text](<SocialComp Poster-1.png>)