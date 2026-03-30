from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
from numba import njit
from scipy.stats import entropy


@njit
def find_edges_above_threshold(similarity_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """
    Identify edges (i, j) with weights above a threshold in the similarity matrix using numba.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix where each entry represents
                                        the similarity between two molecules.
        threshold (float): Similarity threshold above which an edge is created between nodes.

    Returns:
        List[Tuple[int, int, float]]: List of edges with nodes (i, j) and weights above the threshold.
    """
    edges = []
    num_molecules = similarity_matrix.shape[0]
    for i in range(num_molecules):
        for j in range(i + 1, num_molecules):
            if similarity_matrix[i, j] >= threshold:
                edges.append((i, j, similarity_matrix[i, j]))
    return edges


def build_network_from_similarity(similarity_matrix: np.ndarray, cids: list[str], threshold: float) -> nx.Graph:
    """
    Build a NetworkX graph from a similarity matrix based on a given similarity threshold.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix where each entry represents
                                        the similarity between two molecules.
        cids (List[str]): List of unique identifiers (IDs) for the molecules.
        threshold (float): Similarity threshold above which an edge is created between nodes.

    Returns:
        nx.Graph: A NetworkX graph where nodes represent molecules (identified by IDs) and edges
                  represent pairwise similarities above the threshold.
    """
    # Create a NetworkX graph and add nodes for each molecule by ID
    G = nx.Graph()
    for idx, mol_id in enumerate(cids):
        G.add_node(mol_id)  # Use ID as the node identifier

    # Use numba-accelerated function to find edges above the threshold
    edges = find_edges_above_threshold(similarity_matrix, threshold)

    # Map edges from indices to molecule IDs
    edges_with_ids = [(cids[i], cids[j], weight) for i, j, weight in edges]

    # Add edges to the graph
    G.add_weighted_edges_from(edges_with_ids)
    return G


def generate_networks_for_thresholds(
    similarity_matrix: np.ndarray, cids: list[str], thresholds: list[float]
) -> dict[float, nx.Graph]:
    """
    Generate a dictionary of similarity networks for each specified threshold.

    Args:
        similarity_matrix (np.ndarray): NumPy array for fingerprint similarity.
        cids (List[str]): List of compound ids.
        thresholds (List[float]): List of similarity thresholds to apply.

    Returns:
        Dict[float, nx.Graph]: A dictionary where keys are thresholds and values are the
                               corresponding similarity networks.
    """

    networks = {}
    for threshold in thresholds:
        G = build_network_from_similarity(similarity_matrix, cids, threshold)
        networks[threshold] = G
    return networks


# Function to calculate metrics for a given network
def calculate_network_metrics(G: nx.Graph, name: str) -> dict[str, Any]:
    """
    Calculate various network diversity metrics for a given graph.

    Args:
        G (nx.Graph): The NetworkX graph for which metrics are calculated.
        name (str): The name of the network for identification.

    Returns:
        Dict[str, Any]: A dictionary containing the network name and calculated metrics.
    """
    # 1. Network Density: Proportion of edges to possible edges, measuring overall connectivity
    density = nx.density(G)

    # 2. Modularity: Measures the strength of division of the graph into communities # TODO takes to long
    # communities = list(greedy_modularity_communities(G))
    # modularity = nx.algorithms.community.quality.modularity(G, communities)

    # 3. Clustering Coefficient: Measures the likelihood that neighbors of a node are also connected # TODO takes to long
    # clustering_coefficient = nx.average_clustering(G)

    # 4. Degree Centrality Distribution (std dev): Standard deviation of degree centrality,
    #    reflecting variability in node connectivity
    degree_centrality = nx.degree_centrality(G)
    degree_centrality_std = np.std(list(degree_centrality.values()))

    # 5. Assortativity Coefficient: Measures tendency of nodes to connect with similar nodes (homophily)
    assortativity_coefficient = nx.degree_assortativity_coefficient(G)

    # 6. Network Entropy: Shannon entropy based on degree distribution, indicating structural diversity
    degree_sequence = [d for n, d in G.degree()]
    degree_counts = np.array(list(Counter(degree_sequence).values()))
    degree_probabilities = degree_counts / degree_counts.sum()
    network_entropy = entropy(degree_probabilities)

    # Compile metrics into a dictionary
    return {
        "Network": name,
        "Density": density,
        #   "Modularity": modularity,  # TODO takes to long
        # "Clustering Coefficient": clustering_coefficient,  # TODO takes to long
        "Degree Centrality Std Dev": degree_centrality_std,
        "Assortativity Coefficient": assortativity_coefficient,
        "Network Entropy": network_entropy,
    }
