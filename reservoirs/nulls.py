"""Null-model connectomes for the connectome-RC baseline.

`rewire_degree_preserving` produces a degree-preserving randomized version of a connectome
(a Maslov-Sneppen double-edge-swap on the binary topology, with the original edge weights
re-assigned). Comparing the real connectome against an ensemble of these nulls is the standard
test of whether the empirical topology, not just its degree and weight distribution, carries the
benefit. It is the experiment any connectome reservoir-computing claim ultimately rests on
(cf. Suárez et al. 2024). See the null-model section of examples/combined_examples.ipynb.
"""
import numpy as np
import networkx as nx


def rewire_degree_preserving(A, seed: int = 0, n_swaps_per_edge: int = 10) -> np.ndarray:
    """Return a degree-preserving rewired null of the symmetric weighted adjacency `A`.

    Preserves: the binary degree sequence (exactly) and the multiset of edge weights.
    Randomizes: which node pairs are connected. Self-loops are excluded.
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    G = nx.from_numpy_array((A != 0).astype(int))
    G.remove_edges_from(nx.selfloop_edges(G))
    m = G.number_of_edges()
    if m > 1:
        nx.double_edge_swap(G, nswap=n_swaps_per_edge * m, max_tries=n_swaps_per_edge * m * 20, seed=seed)

    # re-assign the original edge weights (shuffled) onto the rewired topology
    rng = np.random.default_rng(seed)
    weights = A[np.triu_indices(N, 1)]
    weights = weights[weights != 0]
    rng.shuffle(weights)

    edges = list(G.edges())
    if len(edges) != len(weights):  # guard: avoid silently truncating via zip
        raise ValueError(
            f"rewire: edge count ({len(edges)}) != weight count ({len(weights)}); "
            "input must be a symmetric adjacency with zero diagonal."
        )
    B = np.zeros((N, N), dtype=float)
    for (u, v), w in zip(edges, weights):
        B[u, v] = w
        B[v, u] = w
    return B
