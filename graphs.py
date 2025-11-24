# microviral/graphs.py

import networkx as nx
from itertools import combinations
from typing import Tuple

import pandas as pd
import community as community_louvain  # python-louvain


def build_global_user_graph(nodes_df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected user–user graph where users are connected if they
    comment in the same cascade.
    """
    G = nx.Graph()

    grouped = nodes_df.groupby("submission_id")

    for sub_id, group in grouped:
        commenters = group[~group["is_submission"]]["author"].dropna().unique()
        for u, v in combinations(commenters, 2):
            if u != v:
                G.add_edge(u, v)

    print(
        "Global user graph:",
        G.number_of_nodes(), "nodes,",
        G.number_of_edges(), "edges",
    )
    return G


def detect_communities(G: nx.Graph) -> dict:
    """Run Louvain community detection and return user -> community_id mapping."""
    print("Running Louvain community detection…")
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, "community")
    print("Detected", len(set(partition.values())), "communities")
    return partition


def attach_communities_to_nodes(nodes_df: pd.DataFrame, partition: dict) -> pd.DataFrame:
    """Add a 'community_id' column to nodes_df based on author."""
    def map_comm(author):
        if author in partition:
            return partition[author]
        return None

    nodes_df = nodes_df.copy()
    nodes_df["community_id"] = nodes_df["author"].apply(map_comm)
    return nodes_df
