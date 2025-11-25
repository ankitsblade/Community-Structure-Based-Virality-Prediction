# 🧵 Community-Structure-Based Virality Prediction

Predicting whether a Reddit post becomes **micro-viral** or **viral** using:

* Multi-subreddit cascade extraction
* Community structure analysis
* Early cascade dynamics
* Classical ML models
* Graph Neural Networks (GraphSAGE)

This repository provides the full, modular pipeline for collection, processing, analysis, and visualization.

---

## 📁 Project Structure

```
Community-Structure-Based-Virality-Prediction/
│
├── collect_cascades.py       # Multi-subreddit Reddit data collection
├── run_pipeline.py           # Feature engineering + classical ML analysis
├── run_gnn.py                # GraphSAGE training (optional)
│
├── microviral/
│   ├── config.py             # Subreddit list, constants, thresholds
│   ├── logger.py             # Logging setup
│   ├── reddit_client.py      # PRAW client + submission fetcher
│   ├── cascades.py           # Post → cascade node extraction
│   ├── graphs.py             # User graph + Louvain communities
│   ├── features.py           # Feature extraction + virality labeling
│   ├── models_ml.py          # Logistic Regression + Random Forest
│   ├── visuals.py            # All plots
│   ├── data_io.py            # Save/load utilities
│   └── gnn.py                # GraphSAGE architecture + dataset builder
│
├── data/                     # Cached cascade datasets
└── figures/                  # Generated visualizations
```

---

## 🚀 Quick Start

### 1️⃣ Create the virtual environment with **uv**

```bash
uv sync
```

This:

* Creates `.venv/`
* Installs **all dependencies** listed in `pyproject.toml`
* Sets up the environment exactly as needed
  No manual `pip install` steps are required.

---

### 2️⃣ Configure Reddit API

Create a `.env` file in the project root:

```
CLIENT_ID=...
CLIENT_SECRET=...
USERNAME=...
PASSWORD=...
USER_AGENT=microviral_app:v1.0
```

---

## 🕸️ Data Collection (Multi-Subreddit)

Fetch cascades across configured subreddits:

```bash
python collect_cascades.py
```

Output:

```
data/nodes_multi.parquet
```

---

## 🧮 Classical ML + Community Analysis Pipeline

```bash
python run_pipeline.py
```

This script:

* Builds global user graph
* Detects Louvain communities
* Extracts cascade + community features
* Labels cascades (micro-viral vs viral)
* Trains baseline ML models
* Saves all plots to `figures/`

---

## 🧠 Graph Neural Network (Optional)

```bash
python run_gnn.py
```

* Builds PyTorch Geometric dataset
* Trains GraphSAGE on cascade graphs
* Reports train/val/test ROC-AUC

---

## 🧩 What the Pipeline Does (Short)

* Collect Reddit cascades
* Build unified node-level dataset
* Model early community structure
* Extract features & label cascades
* Train models (LogReg, RF, GNN)
* Compare performance across subreddits
* Generate ROC curves, feature importance, and cascade visuals

---

## 📊 Example Outputs

Saved to `figures/`:

* Per-subreddit ROC curves
* Virality model comparison curves
* Feature importance plots
* Cascade graphs
* Mock result generators for poster creation

---

## 📜 License

MIT License.

