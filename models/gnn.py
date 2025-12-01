# microviral/gnn.py

from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.explain import GNNExplainer

from config import RANDOM_STATE
from logger import logger

# Optional Weights & Biases logging
try:
    import wandb
except ImportError:
    wandb = None


# ------------------------------------------------------------
# 1. Build per-cascade graphs (Data objects)
# ------------------------------------------------------------

def _build_single_cascade_graph(
    sub_nodes: pd.DataFrame,
    label: int,
    subreddit: Optional[str] = None,
    feature_config: str = "full",
) -> Data:
    """
    Build a PyG Data object for a single cascade.

    Nodes: all submission + comments in that cascade.
    Edges: parent -> child reply edges (bidirectional).

    Node features (base components):
      - depth_norm: depth / max_depth
      - score_log: log1p(max(score, 0))
      - is_root: 1.0 if submission, 0.0 otherwise
      - time_norm: (created_utc - min_time) / (max_time - min_time + eps)

    feature_config:
      - "full": [depth_norm, score_log, is_root, time_norm]
      - "structure_only": [depth_norm, is_root]
      - "structure_time": [depth_norm, is_root, time_norm]
    """

    sub_nodes = sub_nodes.copy()

    # Map node_id -> local index
    node_ids = sub_nodes["node_id"].tolist()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # Base features
    depths = sub_nodes["depth"].fillna(0).to_numpy(dtype=np.float32)
    max_depth = depths.max() if len(depths) > 0 else 0.0
    depth_norm = depths / max_depth if max_depth > 0 else np.zeros_like(depths)

    scores = sub_nodes["score"].fillna(0).to_numpy(dtype=np.float32)
    score_log = np.log1p(np.clip(scores, a_min=0.0, a_max=None))

    is_root = sub_nodes["is_submission"].astype(float).to_numpy(dtype=np.float32)

    times = sub_nodes["created_utc"].fillna(sub_nodes["created_utc"].min()).to_numpy(
        dtype=np.float64
    )
    t_min = times.min() if len(times) > 0 else 0.0
    t_rel = times - t_min
    t_max = t_rel.max() if len(t_rel) > 0 else 0.0
    eps = 1e-8
    time_norm = t_rel / (t_max + eps) if t_max > 0 else np.zeros_like(t_rel)

    # Assemble feature matrix based on feature_config
    if feature_config == "structure_only":
        feats = [depth_norm, is_root]
    elif feature_config == "structure_time":
        feats = [depth_norm, is_root, time_norm]
    else:  # "full"
        feats = [depth_norm, score_log, is_root, time_norm]

    x = np.stack(feats, axis=1).astype(np.float32)  # [num_nodes, num_features]
    x = torch.tensor(x, dtype=torch.float32)

    # Edges: parent -> child for any node with parent_id present in this cascade
    edges_src = []
    edges_dst = []
    for _, row in sub_nodes.iterrows():
        parent_id = row["parent_id"]
        child_id = row["node_id"]
        if pd.notna(parent_id) and parent_id in id2idx:
            p_idx = id2idx[parent_id]
            c_idx = id2idx[child_id]
            # Bidirectional edges
            edges_src.append(p_idx)
            edges_dst.append(c_idx)
            edges_src.append(c_idx)
            edges_dst.append(p_idx)

    if len(edges_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    # Attach metadata
    data.subreddit = subreddit
    # submission_id will be attached in build_gnn_dataset
    return data


def build_gnn_dataset(
    nodes_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    feature_config: str = "full",
) -> List[Data]:
    """
    Build a list of PyG Data objects, one per labeled cascade.

    Expects:
      - nodes_df: node-level info (with node_id, parent_id, submission_id, etc.)
      - labeled_df: one row per submission with columns: submission_id, label, subreddit

    Each Data object gets:
      - x, edge_index, y
      - .subreddit
      - .submission_id
    """
    dataset: List[Data] = []

    # Ensure labeled_df is indexed by submission_id for fast lookup
    if "submission_id" in labeled_df.columns:
        labeled_df = labeled_df.set_index("submission_id")
    else:
        raise ValueError("labeled_df must contain 'submission_id' column.")

    for sub_id, row in labeled_df.iterrows():
        label = int(row["label"])
        subreddit = row.get("subreddit", None)
        sub_nodes = nodes_df[nodes_df["submission_id"] == sub_id]

        if sub_nodes.empty:
            continue

        data = _build_single_cascade_graph(
            sub_nodes=sub_nodes,
            label=label,
            subreddit=subreddit,
            feature_config=feature_config,
        )
        data.submission_id = sub_id
        dataset.append(data)

    logger.info(
        f"GNN dataset built with {len(dataset)} cascades "
        f"(feature_config='{feature_config}')."
    )
    return dataset


# ------------------------------------------------------------
# 2. GraphSAGE model
# ------------------------------------------------------------

class CascadeGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.lin = nn.Linear(hidden_dim, 1)  # binary classification (logit)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        logits = self.lin(x).view(-1)   # [num_graphs]
        return logits


# ------------------------------------------------------------
# 3. Training / evaluation helpers
# ------------------------------------------------------------

def _split_dataset(
    dataset: List[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Data], List[Data], List[Data]]:
    rng = np.random.RandomState(RANDOM_STATE)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def select(idxs):
        return [dataset[i] for i in idxs]

    return select(train_idx), select(val_idx), select(test_idx)


def _epoch_run(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[Adam],
    device: torch.device,
    train: bool = True,
) -> Tuple[float, float]:
    """
    Returns:
      - average loss
      - ROC-AUC (if possible)
    """
    if train:
        model.train()
    else:
        model.eval()

    all_losses = []
    all_probs = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.float())

        if train:
            loss.backward()
            optimizer.step()

        all_losses.append(loss.item())
        all_labels.append(batch.y.detach().cpu().numpy())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    if len(all_losses) == 0:
        return float("nan"), float("nan")

    avg_loss = float(np.mean(all_losses))

    try:
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_probs)
        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")

    return avg_loss, auc


def _train_on_dataset(
    dataset: List[Data],
    experiment_name: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 50,
    use_wandb: bool = False,
) -> Dict[str, float]:
    """
    Train a CascadeGNN on a given dataset with internal train/val/test split.
    Returns dict with train/val/test AUCs and per-epoch history.
    """
    if len(dataset) < 10:
        logger.warning(
            f"[{experiment_name}] Not enough cascades ({len(dataset)}) for GNN training."
        )
        return {
            "train_auc": np.nan,
            "val_auc": np.nan,
            "test_auc": np.nan,
            "history": {},
        }

    train_set, val_set, test_set = _split_dataset(dataset)

    logger.info(
        f"[{experiment_name}] Dataset split: "
        f"{len(train_set)} train / {len(val_set)} val / {len(test_set)} test graphs."
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    in_channels = dataset[0].x.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{experiment_name}] Using device: {device}")

    model = CascadeGNN(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
    }

    wandb_run = None
    if use_wandb and wandb is not None:
        logger.info(f"[{experiment_name}] Initializing Weights & Biases run.")
        wandb_run = wandb.init(
            project="microviral-gnn",
            name=experiment_name,
            config={
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "lr": lr,
                "max_epochs": max_epochs,
                "experiment_name": experiment_name,
            },
            reinit=True,
        )

    train_auc = float("nan")  # for safety in case loop never runs

    for epoch in range(1, max_epochs + 1):
        train_loss, train_auc = _epoch_run(model, train_loader, optimizer, device, train=True)
        val_loss, val_auc = _epoch_run(model, val_loader, optimizer, device, train=False)

        logger.info(
            f"[{experiment_name}] Epoch {epoch:03d} | "
            f"Train loss={train_loss:.4f}, AUC={train_auc:.4f} | "
            f"Val loss={val_loss:.4f}, AUC={val_auc:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                }
            )

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if wandb_run is not None:
        wandb_run.finish()

    if best_state is not None:
        logger.info(f"[{experiment_name}] Loading best model (val AUC={best_val_auc:.4f})…")
        model.load_state_dict(best_state)

    test_loss, test_auc = _epoch_run(model, test_loader, optimizer, device, train=False)
    logger.info(
        f"[{experiment_name}] Final Test Loss={test_loss:.4f}, Test ROC-AUC={test_auc:.4f}"
    )

    return {
        "train_auc": float(train_auc),
        "val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "history": history,
    }


# ------------------------------------------------------------
# 4. Public entrypoints: global, per-subreddit, cross-subreddit, ablations
# ------------------------------------------------------------

def train_gnn_global(
    nodes_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    feature_config: str = "full",
    **kwargs,
) -> Dict[str, float]:
    """
    Train a GNN on cascades from all subreddits combined.
    """
    logger.info(f"[GLOBAL] Building dataset (feature_config='{feature_config}')…")
    dataset = build_gnn_dataset(nodes_df, labeled_df, feature_config=feature_config)
    metrics = _train_on_dataset(dataset, experiment_name=f"GLOBAL_{feature_config}", **kwargs)
    return metrics


# Backward-compatible alias
train_gnn = train_gnn_global


def train_gnn_per_subreddit(
    nodes_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    feature_config: str = "full",
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Train separate GNN models per subreddit.

    Returns:
      {subreddit: {train_auc, val_auc, test_auc, history}}
    """
    if "subreddit" not in labeled_df.columns:
        logger.warning("[PER-SUB] 'subreddit' column missing; skipping per-subreddit GNN.")
        return {}

    logger.info(f"[PER-SUB] Building shared dataset (feature_config='{feature_config}')…")
    dataset_all = build_gnn_dataset(nodes_df, labeled_df, feature_config=feature_config)

    # Group Data objects by subreddit
    by_sub: Dict[str, List[Data]] = {}
    for data in dataset_all:
        sub = getattr(data, "subreddit", None)
        if sub is None:
            continue
        by_sub.setdefault(sub, []).append(data)

    results: Dict[str, Dict[str, float]] = {}

    for sub, ds in by_sub.items():
        logger.info(f"[PER-SUB] Training GNN for r/{sub} on {len(ds)} cascades…")
        metrics = _train_on_dataset(
            ds,
            experiment_name=f"SUB_{sub}_{feature_config}",
            **kwargs,
        )
        results[sub] = metrics

    return results


def train_gnn_cross_subreddit(
    nodes_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    feature_config: str = "full",
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-subreddit generalization:
      - For each subreddit S:
          - Train/val on cascades from all OTHER subreddits
          - Test on cascades from S
    Returns:
      {heldout_subreddit: {train_auc, val_auc, test_auc, history}}
    """
    if "subreddit" not in labeled_df.columns:
        logger.warning("[X-SUB] 'subreddit' column missing; skipping cross-subreddit GNN.")
        return {}

    logger.info(f"[X-SUB] Building shared dataset (feature_config='{feature_config}')…")
    dataset_all = build_gnn_dataset(nodes_df, labeled_df, feature_config=feature_config)

    # Group by subreddit
    by_sub: Dict[str, List[Data]] = {}
    for data in dataset_all:
        sub = getattr(data, "subreddit", None)
        if sub is None:
            continue
        by_sub.setdefault(sub, []).append(data)

    all_subs = sorted(by_sub.keys())
    results: Dict[str, Dict[str, float]] = {}

    hidden_dim = kwargs.get("hidden_dim", 64)
    num_layers = kwargs.get("num_layers", 2)
    dropout = kwargs.get("dropout", 0.3)
    batch_size = kwargs.get("batch_size", 32)
    lr = kwargs.get("lr", 1e-3)
    max_epochs = kwargs.get("max_epochs", 50)
    use_wandb = kwargs.get("use_wandb", False)

    for heldout in all_subs:
        test_set = by_sub[heldout]
        train_val_set: List[Data] = [
            d for s, ds in by_sub.items() if s != heldout for d in ds
        ]

        if len(test_set) < 5 or len(train_val_set) < 10:
            logger.warning(
                f"[X-SUB] Skipping held-out r/{heldout}: "
                f"train_val={len(train_val_set)}, test={len(test_set)} too small."
            )
            continue

        logger.info(
            f"[X-SUB] Held-out r/{heldout}: "
            f"{len(train_val_set)} train/val, {len(test_set)} test cascades."
        )

        # Split train_val into train/val
        train_set, val_set, _ = _split_dataset(train_val_set, train_ratio=0.8, val_ratio=0.2)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        in_channels = dataset_all[0].x.size(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        experiment_name = f"X_SUB_HELDOUT_{heldout}_{feature_config}"
        logger.info(f"[{experiment_name}] Using device: {device}")

        model = CascadeGNN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr)

        best_val_auc = -1.0
        best_state = None

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }

        wandb_run = None
        if use_wandb and wandb is not None:
            logger.info(f"[{experiment_name}] Initializing Weights & Biases run.")
            wandb_run = wandb.init(
                project="microviral-gnn",
                name=experiment_name,
                config={
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_epochs": max_epochs,
                    "heldout_subreddit": heldout,
                    "experiment_name": experiment_name,
                },
                reinit=True,
            )

        train_auc = float("nan")

        for epoch in range(1, max_epochs + 1):
            train_loss, train_auc = _epoch_run(model, train_loader, optimizer, device, train=True)
            val_loss, val_auc = _epoch_run(model, val_loader, optimizer, device, train=False)

            logger.info(
                f"[{experiment_name}] Epoch {epoch:03d} | "
                f"Train loss={train_loss:.4f}, AUC={train_auc:.4f} | "
                f"Val loss={val_loss:.4f}, AUC={val_auc:.4f}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)

            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_auc": train_auc,
                        "val_auc": val_auc,
                    }
                )

            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict()

        if wandb_run is not None:
            wandb_run.finish()

        if best_state is not None:
            logger.info(f"[{experiment_name}] Loading best model (val AUC={best_val_auc:.4f})…")
            model.load_state_dict(best_state)

        # Evaluate on held-out subreddit
        test_loss, test_auc = _epoch_run(model, test_loader, optimizer, device, train=False)
        logger.info(
            f"[{experiment_name}] Held-out r/{heldout}: Test Loss={test_loss:.4f}, "
            f"Test ROC-AUC={test_auc:.4f}"
        )

        results[heldout] = {
            "train_auc": float(train_auc),
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "history": history,
        }

    return results


def train_gnn_ablation(
    nodes_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    feature_configs: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """
    Simple feature ablation:
      - Run global GNN training with different feature_config settings.

    feature_configs examples:
      ["full", "structure_only", "structure_time"]
    """
    if feature_configs is None:
        feature_configs = ["full", "structure_only", "structure_time"]

    results: Dict[str, Dict[str, float]] = {}
    for cfg in feature_configs:
        logger.info(f"[ABLATION] Running global GNN with feature_config='{cfg}'…")
        metrics = train_gnn_global(nodes_df, labeled_df, feature_config=cfg, **kwargs)
        results[cfg] = metrics
    return results


# ------------------------------------------------------------
# 5. Simple GNNExplainer helper (optional interpretability)
# ------------------------------------------------------------

def explain_cascade(
    model: CascadeGNN,
    data: Data,
    epochs: int = 200,
):
    """
    Run GNNExplainer on a single cascade graph.

    NOTE: This is a minimal helper; you still need to interpret the results
    yourself. It returns (node_mask, edge_mask) and logs a few basic stats.
    """
    device = next(model.parameters()).device
    model.eval()

    explainer = GNNExplainer(model, epochs=epochs)
    data = data.to(device)
    node_feat_mask, edge_mask = explainer.explain_graph(
        x=data.x,
        edge_index=data.edge_index,
        batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device),
    )

    logger.info(
        f"[EXPLAIN] Explained cascade submission_id={getattr(data, 'submission_id', 'N/A')}, "
        f"subreddit={getattr(data, 'subreddit', 'N/A')}"
    )
    logger.info(f"[EXPLAIN] Node feature mask shape: {node_feat_mask.shape}")
    logger.info(f"[EXPLAIN] Edge mask shape: {edge_mask.shape}")

    return node_feat_mask, edge_mask
