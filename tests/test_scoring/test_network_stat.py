import numpy as np
import pytest

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NX_AVAILABLE, reason="networkx not importable")


@pytest.fixture
def _network_imports():
    """Lazy import of network_stat functions."""
    from src.cdr_bench.scoring.network_stat import (
        build_network_from_similarity,
        calculate_network_metrics,
        generate_networks_for_thresholds,
    )

    return build_network_from_similarity, calculate_network_metrics, generate_networks_for_thresholds


@pytest.mark.requires_numba
class TestBuildNetworkFromSimilarity:
    def test_creates_graph(self, _network_imports):
        build_network_from_similarity, _, _ = _network_imports
        sim = np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.6],
                [0.3, 0.6, 1.0],
            ]
        )
        cids = ["mol_0", "mol_1", "mol_2"]
        G = build_network_from_similarity(sim, cids, threshold=0.5)
        assert isinstance(G, nx.Graph)
        assert len(G.nodes) == 3

    def test_correct_edges(self, _network_imports):
        build_network_from_similarity, _, _ = _network_imports
        sim = np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.6],
                [0.3, 0.6, 1.0],
            ]
        )
        cids = ["A", "B", "C"]
        G = build_network_from_similarity(sim, cids, threshold=0.5)
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")
        assert not G.has_edge("A", "C")

    def test_all_below_threshold(self, _network_imports):
        build_network_from_similarity, _, _ = _network_imports
        sim = np.array(
            [
                [1.0, 0.2, 0.1],
                [0.2, 1.0, 0.3],
                [0.1, 0.3, 1.0],
            ]
        )
        cids = ["A", "B", "C"]
        G = build_network_from_similarity(sim, cids, threshold=0.99)
        assert len(G.edges) == 0


@pytest.mark.requires_numba
class TestGenerateNetworksForThresholds:
    def test_returns_dict_of_graphs(self, _network_imports):
        _, _, generate_networks_for_thresholds = _network_imports
        sim = np.array(
            [
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.6],
                [0.3, 0.6, 1.0],
            ]
        )
        cids = ["A", "B", "C"]
        networks = generate_networks_for_thresholds(sim, cids, [0.5, 0.7])
        assert len(networks) == 2
        assert 0.5 in networks
        assert 0.7 in networks
        assert isinstance(networks[0.5], nx.Graph)


class TestCalculateNetworkMetrics:
    def test_returns_expected_keys(self, _network_imports):
        _, calculate_network_metrics, _ = _network_imports
        G = nx.complete_graph(5)
        metrics = calculate_network_metrics(G, "test_network")
        assert "Network" in metrics
        assert "Density" in metrics
        assert "Degree Centrality Std Dev" in metrics
        assert "Assortativity Coefficient" in metrics
        assert "Network Entropy" in metrics

    def test_complete_graph_density(self, _network_imports):
        _, calculate_network_metrics, _ = _network_imports
        G = nx.complete_graph(5)
        metrics = calculate_network_metrics(G, "K5")
        assert metrics["Density"] == pytest.approx(1.0)

    def test_path_graph(self, _network_imports):
        _, calculate_network_metrics, _ = _network_imports
        G = nx.path_graph(10)
        metrics = calculate_network_metrics(G, "path")
        assert 0 < metrics["Density"] < 1

    def test_empty_graph_density(self, _network_imports):
        _, calculate_network_metrics, _ = _network_imports
        G = nx.Graph()
        G.add_nodes_from(range(5))
        metrics = calculate_network_metrics(G, "empty")
        assert metrics["Density"] == pytest.approx(0.0)
