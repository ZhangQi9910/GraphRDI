"""RNA secondary-structure graph construction utilities.

This unified file keeps two RNA-graph construction modes from the uploaded code:

1. short/mirna mode: follows the original miRNA RNA_2D.py settings, including
   score-matrix construction, free-energy checking, non-crossing/non-overlap
   constraints, and backbone + secondary-pair adjacency.
2. long/circrna/lncrna mode: follows the original RNA_2D(1).py long-RNA
   settings, including sequence segmentation, overlap, local pair search,
   consecutive-stem scoring, non-crossing/non-overlap filtering, and adjacency
   construction.

The public entry point is build_rna_graph(seq, max_nodes, mode). Use
mode="mirna" for miRNA and mode="long" for circRNA/lncRNA.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Pair = Tuple[int, int]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def normalize_rna_sequence(seq: str, max_len: int | None = None) -> str:
    """Uppercase, convert T to U, retain A/U/C/G/N, and optionally truncate."""
    seq = str(seq).upper().replace("T", "U").strip()
    seq = "".join(base for base in seq if base in {"A", "U", "C", "G", "N"})
    if max_len is not None:
        seq = seq[:max_len]
    return seq


def sequence_to_onehot(seq: str) -> np.ndarray:
    """Encode A/U/C/G bases as a 4-dimensional one-hot matrix. N stays all-zero."""
    base_to_idx = {"A": 0, "U": 1, "C": 2, "G": 3}
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for idx, base in enumerate(seq):
        if base in base_to_idx:
            onehot[idx, base_to_idx[base]] = 1.0
    return onehot


def build_adjacency_matrix(seq_length: int, secondary_pairs: Iterable[Pair]) -> np.ndarray:
    """Build an RNA adjacency matrix with backbone edges and secondary-pair edges."""
    adj = np.zeros((seq_length, seq_length), dtype=np.float32)
    for i in range(seq_length - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    for i, j in secondary_pairs:
        if 0 <= i < seq_length and 0 <= j < seq_length and i != j:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


# ---------------------------------------------------------------------------
# miRNA / short-RNA mode: original RNA_2D.py settings
# ---------------------------------------------------------------------------
def paired(x: str, y: str) -> float:
    """Base-pairing score used by the original miRNA RNA_2D.py."""
    pairs = {
        ("A", "U"): 2.0,
        ("U", "A"): 2.0,
        ("G", "C"): 3.0,
        ("C", "G"): 3.0,
        ("G", "U"): 0.8,
        ("U", "G"): 0.8,
    }
    return pairs.get((x, y), 0.0)


def Gaussian(x: int) -> float:
    """Gaussian decay used in the original miRNA RNA_2D.py."""
    return math.exp(-0.5 * (x**2))


def get_base_pair_energy(x: str, y: str) -> float:
    """Approximate base-pair free energy from the original miRNA RNA_2D.py."""
    energy = {
        ("A", "U"): -2.4,
        ("U", "A"): -2.4,
        ("G", "C"): -3.4,
        ("C", "G"): -3.4,
        ("G", "U"): -1.3,
        ("U", "G"): -1.3,
    }
    return energy.get((x, y), 5.0)


def create_score_matrix_mirna(seq: str) -> np.ndarray:
    """Original miRNA score-matrix construction from RNA_2D.py."""
    seq = normalize_rna_sequence(seq)
    length = len(seq)
    score_matrix = np.zeros((length, length), dtype=np.float32)

    for i in range(length):
        # Original setting: search within i +/- 50.
        for j in range(max(0, i - 50), min(length, i + 50)):
            if i == j:
                continue

            coefficient = 0.0
            # Forward search: i-add and j+add.
            for add in range(30):
                if (i - add >= 0) and (j + add < length):
                    score = paired(seq[i - add], seq[j + add])
                    if score == 0:
                        break
                    coefficient += score * Gaussian(add)
                else:
                    break

            # Reverse search: i+add and j-add.
            if coefficient > 0:
                for add in range(1, 30):
                    if (i + add < length) and (j - add >= 0):
                        score = paired(seq[i + add], seq[j - add])
                        if score == 0:
                            break
                        coefficient += score * Gaussian(add)
                    else:
                        break

            energy = get_base_pair_energy(seq[i], seq[j])
            if energy < 0:
                coefficient *= abs(energy)
            score_matrix[i, j] = coefficient

    return score_matrix


def get_loop_energy(loop_length: int) -> float:
    """Loop-energy penalty from the original RNA_2D.py."""
    if loop_length < 3:
        return 2.0
    if 3 <= loop_length <= 8:
        return 1.0 + 0.1 * (loop_length - 3)
    return 1.5 + 0.05 * (loop_length - 8)


def is_non_crossing(pairs: Sequence[Pair], new_i: int, new_j: int) -> bool:
    """Forbid crossing and overlapping pairs."""
    used_pos = {pos for pair in pairs for pos in pair}
    if new_i in used_pos or new_j in used_pos:
        return False
    for i, j in pairs:
        if (i < new_i < j < new_j) or (new_i < i < new_j < j):
            return False
    return True


def calculate_total_energy(seq: str, pairs: Sequence[Pair]) -> float:
    """Total free-energy heuristic from the original RNA_2D.py."""
    if not pairs:
        return 0.0

    total_energy = 0.0
    length = len(seq)
    paired_pos = {pos for pair in pairs for pos in pair}
    unpaired_pos = [i for i in range(length) if i not in paired_pos]

    for i, j in pairs:
        total_energy += get_base_pair_energy(seq[i], seq[j])

    for i, j in sorted(pairs, key=lambda x: x[0]):
        loop_length = j - i - 1
        total_energy += get_loop_energy(loop_length)

    total_energy += 0.1 * len(unpaired_pos)
    return total_energy


def is_consecutive(pairs: Sequence[Pair], new_i: int, new_j: int) -> bool:
    """Check whether a new pair extends an existing stem."""
    return (new_i - 1, new_j + 1) in pairs


def find_optimal_pairs_mirna(seq: str, score_matrix: np.ndarray) -> List[Pair]:
    """Original miRNA greedy energy-aware pair selection from RNA_2D.py."""
    length = len(seq)
    candidates: list[tuple[float, int, int]] = []

    for i in range(length):
        for j in range(i + 1, length):
            if j - i < 3:
                continue
            if score_matrix[i, j] > 0:
                candidates.append((-float(score_matrix[i, j]), i, j))

    candidates.sort()
    optimal_pairs: list[Pair] = []
    current_energy = calculate_total_energy(seq, optimal_pairs)

    for _, i, j in candidates:
        if is_non_crossing(optimal_pairs, i, j):
            temp_pairs = optimal_pairs + [(i, j)]
            temp_energy = calculate_total_energy(seq, temp_pairs)
            if is_consecutive(optimal_pairs, i, j):
                temp_energy -= 1.0
            if temp_energy < current_energy + 0.5:
                optimal_pairs = temp_pairs
                current_energy = temp_energy

    return optimal_pairs


def build_rna_graph_mirna(seq: str, max_nodes: int = 30) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build an RNA graph using the original miRNA RNA_2D.py settings."""
    seq = normalize_rna_sequence(seq, max_len=max_nodes)
    if len(seq) < 2:
        return None, None
    score_matrix = create_score_matrix_mirna(seq)
    pairs = find_optimal_pairs_mirna(seq, score_matrix)
    return build_adjacency_matrix(len(seq), pairs), sequence_to_onehot(seq)


# ---------------------------------------------------------------------------
# circRNA / lncRNA long-RNA mode: original RNA_2D(1).py settings
# ---------------------------------------------------------------------------
def is_complementary(a: str, b: str) -> int:
    """Complementarity scoring from RNA_2D(1).py: strong=2, weak G-U=1."""
    a, b = a.upper().replace("T", "U"), b.upper().replace("T", "U")
    strong_pairs = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    weak_pairs = {("G", "U"), ("U", "G")}
    if (a, b) in strong_pairs:
        return 2
    if (a, b) in weak_pairs:
        return 1
    return 0


def segment_sequence(seq: str, segment_length: int = 150, overlap: int = 50) -> list[tuple[int, int, str]]:
    """Segment long RNA using the RNA_2D(1).py defaults."""
    segments: list[tuple[int, int, str]] = []
    start = 0
    seq_length = len(seq)
    while start < seq_length:
        end = min(start + segment_length, seq_length)
        segments.append((start, end, seq[start:end]))
        if end == seq_length:
            break
        start = end - overlap
    return segments


def filter_best_non_crossing(pairs: Sequence[tuple[int, int, int, int]]) -> list[Pair]:
    """Filter crossing/overlapping pairs, preserving high-scoring pairs first."""
    if not pairs:
        return []

    non_crossing: list[Pair] = []
    used_pos: set[int] = set()
    for i, j, _score, _count in pairs:
        if i in used_pos or j in used_pos:
            continue
        cross = False
        for ni, nj in non_crossing:
            if (ni < i < nj < j) or (i < ni < j < nj):
                cross = True
                break
        if not cross:
            non_crossing.append((i, j))
            used_pos.add(i)
            used_pos.add(j)
    return non_crossing


def find_better_pairs(
    segment_seq: str,
    min_distance: int = 3,
    max_distance: int = 80,
    min_consecutive: int = 2,
    min_score: int = 3,
) -> list[Pair]:
    """Long-RNA pair detection from RNA_2D(1).py."""
    length = len(segment_seq)
    pairs: list[tuple[int, int, int, int]] = []

    for i in range(length):
        start_j = i + min_distance
        end_j = min(i + max_distance, length)
        for j in range(start_j, end_j):
            current_score = is_complementary(segment_seq[i], segment_seq[j])
            if current_score == 0:
                continue

            consecutive_score = current_score
            consecutive_count = 1
            for k in range(1, 10):
                if i + k >= length or j - k < 0:
                    break
                score = is_complementary(segment_seq[i + k], segment_seq[j - k])
                if score == 0:
                    break
                consecutive_score += score
                consecutive_count += 1

            if (consecutive_count >= min_consecutive) or (consecutive_score >= min_score):
                pairs.append((i, j, consecutive_score, consecutive_count))

    pairs.sort(key=lambda x: (-x[2], -x[3], x[0], x[1]))
    return filter_best_non_crossing(pairs)


def combine_segment_pairs(segments: Sequence[tuple[int, int, str]], segment_pairs_list: Sequence[Sequence[Pair]]) -> list[Pair]:
    """Combine segment-local pair coordinates into original sequence coordinates."""
    all_pairs: list[Pair] = []
    seen: set[Pair] = set()
    for (start, _end, _seg_seq), pairs in zip(segments, segment_pairs_list):
        for i, j in pairs:
            sorted_pair = tuple(sorted((i + start, j + start)))
            if sorted_pair not in seen:
                seen.add(sorted_pair)
                all_pairs.append(sorted_pair)
    return all_pairs


def analyze_rna_long(
    seq: str,
    segment_length: int = 150,
    overlap: int = 50,
    min_distance: int = 3,
    max_distance: int = 80,
    min_consecutive: int = 2,
    min_score: int = 3,
) -> list[Pair]:
    """Long-RNA analysis using the RNA_2D(1).py default settings, without prints."""
    segments = segment_sequence(seq, segment_length=segment_length, overlap=overlap)
    segment_pairs_list = [
        find_better_pairs(
            seg_seq,
            min_distance=min_distance,
            max_distance=max_distance,
            min_consecutive=min_consecutive,
            min_score=min_score,
        )
        for _start, _end, seg_seq in segments
    ]
    return combine_segment_pairs(segments, segment_pairs_list)


def build_rna_graph_long(seq: str, max_nodes: int = 400) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build an RNA graph using the original long-RNA RNA_2D(1).py settings."""
    seq = normalize_rna_sequence(seq, max_len=max_nodes)
    if len(seq) < 2:
        return None, None
    pairs = analyze_rna_long(
        seq,
        segment_length=150,
        overlap=50,
        min_distance=3,
        max_distance=80,
        min_consecutive=2,
        min_score=3,
    )
    pairs = [(i, j) for i, j in pairs if i < len(seq) and j < len(seq)]
    return build_adjacency_matrix(len(seq), pairs), sequence_to_onehot(seq)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def normalize_graph_mode(mode: str | None = None, rna_type: str | None = None) -> str:
    """Normalize graph mode/rna_type into either 'mirna' or 'long'."""
    raw = (mode or rna_type or "mirna").lower().replace("_", "").replace("-", "")
    if raw in {"mirna", "short", "shortrna"}:
        return "mirna"
    if raw in {"circrna", "lncrna", "long", "longrna"}:
        return "long"
    raise ValueError(f"Unsupported RNA graph mode/rna_type: {mode or rna_type}")


def build_rna_graph(
    seq: str,
    max_nodes: int = 400,
    mode: str = "mirna",
    rna_type: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build RNA adjacency matrix and node one-hot features.

    Args:
        seq: RNA sequence.
        max_nodes: maximum sequence length retained in the graph.
        mode: 'mirna' uses the original RNA_2D.py settings; 'long' uses the
            original RNA_2D(1).py settings.
        rna_type: optional alias; circRNA/lncRNA map to 'long'.
    """
    graph_mode = normalize_graph_mode(mode=mode, rna_type=rna_type)
    if graph_mode == "mirna":
        return build_rna_graph_mirna(seq, max_nodes=max_nodes)
    return build_rna_graph_long(seq, max_nodes=max_nodes)
