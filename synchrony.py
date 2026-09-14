# ==========================================================
# synchrony.py
#
# Pairwise functional connectivity and network topology.
#
# Network burst detection answers "when did the network fire together". It
# says nothing about *which* units fire together, and that is where most
# neurodevelopmental phenotypes live: SHANK2 networks are hyperconnected,
# MECP2 and TSC2 networks lose connectivity, and in each case burst rate can
# look unremarkable while the connectivity structure has changed.
#
# Correlation measure: the spike time tiling coefficient (Cutts & Eglen 2014,
# J Neurosci 34:14288-14303). Chosen over the correlation index and plain
# cross-correlation because it is not confounded by firing rate, which matters
# here: a hypoactive genotype would otherwise appear less connected purely
# because its units fire less.
#
# Topology: standard graph metrics on the thresholded STTC matrix, normalised
# against random graphs of the same size and edge count. Ort et al. 2026 show
# that the choice of connectivity method and threshold can change apparent
# network structure as much as real biology does, so every number here is
# reported alongside the window, threshold and null model that produced it.
# ==========================================================

from __future__ import annotations

import numpy as np

try:  # networkx is a hard dependency of the pipeline, but stay importable.
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

# Two synchronicity windows are reported because the answer depends on the
# timescale: 10 ms is roughly monosynaptic, 25 ms captures common drive within
# a burst. Reporting one alone invites reading a timescale effect as a
# connectivity effect.
DEFAULT_WINDOWS_S = (0.010, 0.025)

# Pairwise cost is quadratic in the unit count. Above this many units the
# matrix is computed on a random subsample, which is reported.
DEFAULT_MAX_UNITS = 400


# ---------------------------------------------------------------------------
# Spike time tiling coefficient
# ---------------------------------------------------------------------------

def tiling_fraction(times, dt, t_start, t_stop):
    """Fraction of the recording covered by +/-dt tiles around `times`.

    Overlapping tiles are counted once, and tiles are clipped to the
    recording, so a unit firing near either end is not credited with coverage
    outside it.
    """
    times = np.asarray(times, dtype=float)
    total_time = float(t_stop - t_start)
    if times.size == 0 or total_time <= 0:
        return 0.0

    # Each spike contributes 2*dt, minus whatever it shares with its
    # predecessor.
    covered = 2.0 * dt
    if times.size > 1:
        gaps = np.diff(times)
        covered += float(np.sum(np.minimum(gaps, 2.0 * dt)))

    # Clip the overhang at both ends of the recording.
    covered -= max(0.0, t_start - (times[0] - dt))
    covered -= max(0.0, (times[-1] + dt) - t_stop)

    return float(np.clip(covered / total_time, 0.0, 1.0))


def _proportion_within(times_a, times_b, dt):
    """Fraction of spikes in A that fall within dt of any spike in B."""
    times_a = np.asarray(times_a, dtype=float)
    times_b = np.asarray(times_b, dtype=float)
    if times_a.size == 0 or times_b.size == 0:
        return 0.0

    insert = np.searchsorted(times_b, times_a)
    before = times_b[np.clip(insert - 1, 0, times_b.size - 1)]
    after = times_b[np.clip(insert, 0, times_b.size - 1)]
    nearest = np.minimum(np.abs(times_a - before), np.abs(times_a - after))
    return float(np.count_nonzero(nearest <= dt) / times_a.size)


def _sttc_term(proportion, tiling):
    """One direction of the STTC.

    The denominator vanishes when the tiles of one train already cover the
    whole recording; the numerator vanishes with it, and the limit carries no
    information about coupling, so the term is 0.
    """
    denominator = 1.0 - proportion * tiling
    if abs(denominator) < 1e-12:
        return 0.0
    return (proportion - tiling) / denominator


def spike_time_tiling_coefficient(times_a, times_b, dt, t_start, t_stop,
                                  tiling_a=None, tiling_b=None):
    """STTC of two spike trains. NaN if either train is empty.

    Ranges from -1 (anticorrelated) through 0 (independent) to +1.
    `tiling_a`/`tiling_b` can be passed in when already computed, which is
    what makes the full matrix affordable.
    """
    times_a = np.asarray(times_a, dtype=float)
    times_b = np.asarray(times_b, dtype=float)
    if times_a.size == 0 or times_b.size == 0:
        return float("nan")

    if tiling_a is None:
        tiling_a = tiling_fraction(times_a, dt, t_start, t_stop)
    if tiling_b is None:
        tiling_b = tiling_fraction(times_b, dt, t_start, t_stop)

    proportion_a = _proportion_within(times_a, times_b, dt)
    proportion_b = _proportion_within(times_b, times_a, dt)
    return 0.5 * (_sttc_term(proportion_a, tiling_b) + _sttc_term(proportion_b, tiling_a))


def sttc_matrix(trains, dt, t_start, t_stop):
    """Symmetric STTC matrix with a NaN diagonal."""
    n_units = len(trains)
    matrix = np.full((n_units, n_units), np.nan, dtype=float)
    if n_units == 0:
        return matrix

    tilings = [tiling_fraction(t, dt, t_start, t_stop) for t in trains]
    for i in range(n_units):
        for j in range(i + 1, n_units):
            value = spike_time_tiling_coefficient(
                trains[i], trains[j], dt, t_start, t_stop,
                tiling_a=tilings[i], tiling_b=tilings[j],
            )
            matrix[i, j] = matrix[j, i] = value
    return matrix


# ---------------------------------------------------------------------------
# Significance threshold
# ---------------------------------------------------------------------------

def _circular_shift(times, shift, t_start, t_stop):
    """Rotate a train within the recording, preserving its intervals."""
    total_time = t_stop - t_start
    return np.sort(t_start + np.mod(np.asarray(times, dtype=float) - t_start + shift, total_time))


def null_sttc_threshold(trains, dt, t_start, t_stop, percentile=95.0,
                        n_pairs=500, n_shifts=20, seed=0):
    """STTC value that independent trains rarely exceed.

    Built by circularly shifting one train of a pair, which keeps each train's
    firing rate and interval structure while destroying any real coupling.

    This is a single pooled threshold rather than a per-pair p-value. A per-pair
    null across every pair is what a small electrode array can afford; with
    hundreds of sorted units it is not, and the pooled threshold is the
    approximation that keeps the estimate honest about its own cost. Rate
    heterogeneity makes it approximate, so treat `fraction_significant_pairs`
    as a comparative measure between wells analysed identically, not as an
    absolute count of real connections.
    """
    n_units = len(trains)
    if n_units < 2:
        return None, {"n_null_values": 0}

    rng = np.random.default_rng(seed)
    total_time = t_stop - t_start
    null_values = []

    for _ in range(int(n_pairs)):
        i, j = rng.choice(n_units, size=2, replace=False)
        if len(trains[i]) == 0 or len(trains[j]) == 0:
            continue
        tiling_i = tiling_fraction(trains[i], dt, t_start, t_stop)
        for _ in range(int(n_shifts)):
            shifted = _circular_shift(
                trains[j], rng.uniform(0.0, total_time), t_start, t_stop
            )
            value = spike_time_tiling_coefficient(
                trains[i], shifted, dt, t_start, t_stop, tiling_a=tiling_i,
            )
            if np.isfinite(value):
                null_values.append(value)

    if not null_values:
        return None, {"n_null_values": 0}

    null_values = np.asarray(null_values, dtype=float)
    return float(np.percentile(null_values, percentile)), {
        "n_null_values": int(null_values.size),
        "null_mean": float(null_values.mean()),
        "null_std": float(null_values.std()),
        "percentile": float(percentile),
        "n_shifts": int(n_shifts),
    }


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------

def _clustering_from_adjacency(adjacency, degrees):
    """Per-node clustering coefficient from a binary adjacency matrix.

    Counts triangles as diag(A^3)/2 with two matrix products. networkx does
    the same count in Python, which costs order n*d^2 and becomes the
    dominant term for the dense, few-hundred-unit graphs this pipeline
    produces; the same arithmetic in numpy is three orders of magnitude
    faster and lets the random-graph reference stay affordable.
    """
    squared = adjacency @ adjacency
    triangles = np.einsum("ij,ji->i", squared, adjacency) / 2.0
    possible = degrees * (degrees - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        clustering = np.where(possible > 0, 2.0 * triangles / possible, 0.0)
    return clustering


def _weighted_clustering(weights, degrees):
    """Onnela weighted clustering: triangle intensity, not just count."""
    largest = np.max(weights)
    if largest <= 0:
        return np.zeros(weights.shape[0])
    scaled = np.cbrt(weights / largest)
    squared = scaled @ scaled
    triangles = np.einsum("ij,ji->i", squared, scaled)
    possible = degrees * (degrees - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(possible > 0, triangles / possible, 0.0)


def _path_metrics(adjacency):
    """(characteristic path length, global efficiency).

    Path length is averaged within the largest connected component, the usual
    convention: with any isolated node the whole-graph mean is infinite.
    Efficiency uses every pair, counting unreachable pairs as zero.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path

    graph = csr_matrix(adjacency)
    distances = shortest_path(graph, method="D", unweighted=True, directed=False)

    off_diagonal = ~np.eye(adjacency.shape[0], dtype=bool)
    reachable = np.isfinite(distances) & off_diagonal
    with np.errstate(divide="ignore"):
        inverse = np.where(reachable, 1.0 / distances, 0.0)
    efficiency = float(inverse[off_diagonal].mean()) if off_diagonal.any() else None

    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    members = np.flatnonzero(labels == largest)
    if members.size < 2:
        return None, efficiency, n_components, sizes.max()

    block = distances[np.ix_(members, members)]
    mask = ~np.eye(members.size, dtype=bool)
    path_length = float(block[mask].mean())
    return path_length, efficiency, n_components, sizes.max()


def _random_adjacency(n_nodes, n_edges, rng):
    """Erdos-Renyi graph with exactly n_edges, as an adjacency matrix."""
    n_possible = n_nodes * (n_nodes - 1) // 2
    chosen = rng.choice(n_possible, size=min(n_edges, n_possible), replace=False)
    rows, cols = np.triu_indices(n_nodes, k=1)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[rows[chosen], cols[chosen]] = 1.0
    return adjacency + adjacency.T


def graph_metrics(matrix, threshold, n_random=5, seed=0, max_nodes_for_paths=1500):
    """Topology of the graph formed by thresholding the STTC matrix.

    Clustering and path length are also reported normalised by random graphs
    with the same node and edge count, because both depend strongly on network
    size and density; comparing raw values between wells with different unit
    counts compares the unit counts.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_nodes = matrix.shape[0]
    if n_nodes < 2 or threshold is None:
        return {"n_nodes": int(n_nodes), "n_edges": 0}

    finite = np.isfinite(matrix)
    connected = finite & (matrix >= threshold)
    np.fill_diagonal(connected, False)
    adjacency = connected.astype(float)

    n_edges = int(connected.sum() // 2)
    n_possible = n_nodes * (n_nodes - 1) // 2
    metrics = {
        "n_nodes": int(n_nodes),
        "n_edges": n_edges,
        "density": float(n_edges / n_possible) if n_possible else 0.0,
        "threshold": float(threshold),
    }
    if n_edges == 0:
        return metrics

    degrees = adjacency.sum(axis=1)
    metrics["mean_degree"] = float(degrees.mean())
    metrics["degree_cv"] = (
        float(degrees.std() / degrees.mean()) if degrees.mean() > 0 else None
    )
    # Hubs: nodes far above the typical degree. A rising hub fraction with
    # falling density is a network reorganising around a few units.
    hub_cutoff = degrees.mean() + 2.0 * degrees.std()
    metrics["hub_fraction"] = float(np.count_nonzero(degrees > hub_cutoff) / n_nodes)

    clustering = float(_clustering_from_adjacency(adjacency, degrees).mean())
    metrics["mean_clustering"] = clustering

    weights = np.where(connected, np.nan_to_num(matrix, nan=0.0), 0.0)
    metrics["mean_clustering_weighted"] = float(_weighted_clustering(weights, degrees).mean())
    metrics["mean_sttc_of_edges"] = float(matrix[connected].mean())

    path_length = None
    if n_nodes <= max_nodes_for_paths:
        path_length, efficiency, n_components, largest = _path_metrics(adjacency)
        metrics["characteristic_path_length"] = path_length
        metrics["global_efficiency"] = efficiency
        metrics["n_components"] = int(n_components)
        metrics["largest_component_fraction"] = float(largest / n_nodes)
    else:
        metrics["characteristic_path_length"] = None
        metrics["paths_skipped_n_nodes_above"] = int(max_nodes_for_paths)

    if nx is not None:
        try:
            graph = nx.from_numpy_array(np.where(connected, matrix, 0.0))
            communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
            metrics["modularity"] = float(
                nx.community.modularity(graph, communities, weight="weight")
            )
            metrics["n_modules"] = len(communities)
        except Exception:
            metrics["modularity"] = None
            metrics["n_modules"] = None

    # Random reference with the same size and edge count.
    rng = np.random.default_rng(seed)
    random_clustering, random_paths = [], []
    for _ in range(int(n_random)):
        reference = _random_adjacency(n_nodes, n_edges, rng)
        reference_degrees = reference.sum(axis=1)
        random_clustering.append(
            float(_clustering_from_adjacency(reference, reference_degrees).mean())
        )
        if path_length is not None:
            reference_path, _, _, _ = _path_metrics(reference)
            if reference_path:
                random_paths.append(reference_path)

    if random_clustering and np.mean(random_clustering) > 0:
        metrics["clustering_normalised"] = float(clustering / np.mean(random_clustering))
    if random_paths and path_length:
        metrics["path_length_normalised"] = float(path_length / np.mean(random_paths))

    normalised_clustering = metrics.get("clustering_normalised")
    normalised_path = metrics.get("path_length_normalised")
    if normalised_clustering and normalised_path and normalised_path > 0:
        # Small-worldness: more clustered than random while staying as easy to
        # traverse. Above 1 is the small-world regime.
        metrics["small_world_sigma"] = float(normalised_clustering / normalised_path)

    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _summarise_matrix(matrix):
    upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    values = matrix[upper]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n_pairs": 0, "mean": None, "median": None, "std": None, "p95": None}
    return {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p95": float(np.percentile(values, 95)),
    }


def compute_synchrony(SpikeTimes, duration_s=None, t_start_s=0.0,
                      windows_s=DEFAULT_WINDOWS_S, max_units=DEFAULT_MAX_UNITS,
                      n_null_pairs=500, n_null_shifts=20, percentile=95.0,
                      n_random_graphs=5, seed=0):
    """Pairwise STTC and network topology for one well.

    Returns {"windows": {...}, "matrices": {...}, "params": {...}}. The caller
    is expected to pop "matrices" before writing JSON; they belong in an npz.
    """
    unit_ids = [u for u in SpikeTimes if len(np.asarray(SpikeTimes[u])) > 0]
    if len(unit_ids) < 2:
        return {
            "windows": {},
            "matrices": {},
            "params": {"n_units": len(unit_ids), "reason": "fewer_than_two_active_units"},
        }

    trains = [np.sort(np.asarray(SpikeTimes[u], dtype=float)) for u in unit_ids]
    pooled = np.concatenate(trains)

    if duration_s is not None and float(duration_s) > 0:
        t_start, t_stop = float(t_start_s), float(t_start_s) + float(duration_s)
        duration_source = "recording"
    else:
        t_start, t_stop = float(pooled.min()), float(pooled.max())
        duration_source = "spike_span"
    if t_stop <= t_start:
        return {"windows": {}, "matrices": {},
                "params": {"reason": "zero_duration"}}

    n_subsampled = None
    if max_units and len(unit_ids) > int(max_units):
        # Random rather than by firing rate: picking the most active units
        # would bias every connectivity measure upward.
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(unit_ids), size=int(max_units), replace=False))
        unit_ids = [unit_ids[i] for i in keep]
        trains = [trains[i] for i in keep]
        n_subsampled = int(max_units)

    windows, matrices = {}, {}
    for window in windows_s:
        label = f"{int(round(window * 1000))}ms"
        matrix = sttc_matrix(trains, window, t_start, t_stop)
        threshold, null_info = null_sttc_threshold(
            trains, window, t_start, t_stop, percentile=percentile,
            n_pairs=n_null_pairs, n_shifts=n_null_shifts, seed=seed,
        )

        summary = _summarise_matrix(matrix)
        summary["window_s"] = float(window)
        summary["significance_threshold"] = threshold
        summary["null"] = null_info
        if threshold is not None and summary["n_pairs"]:
            upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            values = matrix[upper]
            values = values[np.isfinite(values)]
            summary["fraction_significant_pairs"] = float(
                np.count_nonzero(values >= threshold) / values.size
            )
        summary["graph"] = graph_metrics(
            matrix, threshold, n_random=n_random_graphs, seed=seed
        )

        windows[label] = summary
        matrices[label] = matrix

    return {
        "windows": windows,
        "matrices": matrices,
        "unit_ids": [str(u) for u in unit_ids],
        "params": {
            "n_units": len(unit_ids),
            "n_units_subsampled_to": n_subsampled,
            "windows_s": list(windows_s),
            "duration_s": t_stop - t_start,
            "duration_source": duration_source,
            "percentile": percentile,
            "n_null_pairs": n_null_pairs,
            "n_null_shifts": n_null_shifts,
            "n_random_graphs": n_random_graphs,
        },
    }
