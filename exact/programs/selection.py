"""Program selection using hierarchical clustering.

Selects a diverse, representative subset of programs from a larger pool
using edit distance as the similarity metric and hierarchical clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Callable
import numpy as np
from tqdm import tqdm

from .edit_distance import (
    ProgramTree,
    parse_to_tree,
    program_edit_distance,
)


@dataclass
class SelectionResult:
    """Result of program selection.
    
    Attributes:
        selected_programs: List of selected program strings
        selected_indices: Indices of selected programs in the original list
        cluster_labels: Cluster assignment for each original program
        cluster_sizes: Number of programs in each cluster
        distance_matrix: Pairwise distance matrix (if computed)
    """
    selected_programs: list[str]
    selected_indices: list[int]
    cluster_labels: np.ndarray
    cluster_sizes: dict[int, int]
    distance_matrix: Optional[np.ndarray] = None
    
    @property
    def num_selected(self) -> int:
        return len(self.selected_programs)
    
    @property
    def num_clusters(self) -> int:
        return len(self.cluster_sizes)
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Selected {self.num_selected} programs from {len(self.cluster_labels)} total",
            f"Number of clusters: {self.num_clusters}",
            f"Cluster sizes: min={min(self.cluster_sizes.values())}, "
            f"max={max(self.cluster_sizes.values())}, "
            f"mean={np.mean(list(self.cluster_sizes.values())):.1f}",
        ]
        return "\n".join(lines)


def compute_distance_matrix(
    programs: list[Union[str, ProgramTree]],
    delta: Optional[Callable] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute pairwise edit distance matrix for programs.
    
    Args:
        programs: List of programs (strings or ProgramTree objects)
        delta: Optional custom cost function for edit distance
        show_progress: Whether to show progress bar
        
    Returns:
        Symmetric distance matrix of shape (n, n)
    """
    # Parse all programs to trees first
    trees = [
        parse_to_tree(p) if isinstance(p, str) else p
        for p in programs
    ]
    n = len(trees)
    
    # Initialize distance matrix
    distances = np.zeros((n, n))
    
    # Compute upper triangle (matrix is symmetric)
    total_pairs = n * (n - 1) // 2
    iterator = range(n)
    
    if show_progress:
        pbar = tqdm(total=total_pairs, desc="Computing distance matrix")
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = program_edit_distance(trees[i], trees[j], delta)
            distances[i, j] = dist
            distances[j, i] = dist
            
            if show_progress:
                pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    return distances


def select_programs_hierarchical(
    programs: list[str],
    budget: int,
    distance_matrix: Optional[np.ndarray] = None,
    linkage_method: str = "average",
    delta: Optional[Callable] = None,
    show_progress: bool = True,
) -> SelectionResult:
    """Select diverse programs using hierarchical clustering.
    
    Uses agglomerative hierarchical clustering with edit distance to group
    similar programs, then selects the medoid (most central program) from
    each cluster to maximize diversity.
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        distance_matrix: Pre-computed distance matrix (optional, computed if None)
        linkage_method: Linkage method for clustering ('average', 'complete', 'single')
        delta: Optional custom cost function for edit distance
        show_progress: Whether to show progress bars
        
    Returns:
        SelectionResult with selected programs and metadata
        
    Note:
        For 10,000 programs, distance matrix computation is O(n²) = 50M pairs.
        With ~1ms per edit distance, this takes ~14 hours. Consider:
        - Using a smaller sample for initial experiments
        - Caching the distance matrix
        - Using approximate methods for very large N
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    n = len(programs)
    
    if budget >= n:
        # Select all programs
        return SelectionResult(
            selected_programs=programs.copy(),
            selected_indices=list(range(n)),
            cluster_labels=np.arange(n),
            cluster_sizes={i: 1 for i in range(n)},
            distance_matrix=distance_matrix,
        )
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        if show_progress:
            print(f"Computing {n}x{n} distance matrix ({n*(n-1)//2:,} pairs)...")
        distance_matrix = compute_distance_matrix(programs, delta, show_progress)
    
    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    if show_progress:
        print(f"Performing hierarchical clustering with {linkage_method} linkage...")
    
    Z = linkage(condensed, method=linkage_method)
    
    # Cut tree to get exactly `budget` clusters
    cluster_labels = fcluster(Z, t=budget, criterion='maxclust')
    
    # Find medoid (most central program) for each cluster
    selected_indices = []
    cluster_sizes = {}
    
    for cluster_id in range(1, budget + 1):
        # Get indices of programs in this cluster
        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_sizes[cluster_id] = len(cluster_indices)
        
        if len(cluster_indices) == 1:
            # Single program in cluster
            medoid_idx = cluster_indices[0]
        else:
            # Find medoid: program with minimum total distance to others in cluster
            sub_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            total_distances = sub_matrix.sum(axis=1)
            medoid_local_idx = np.argmin(total_distances)
            medoid_idx = cluster_indices[medoid_local_idx]
        
        selected_indices.append(medoid_idx)
    
    # Extract selected programs
    selected_programs = [programs[i] for i in selected_indices]
    
    return SelectionResult(
        selected_programs=selected_programs,
        selected_indices=selected_indices,
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
        distance_matrix=distance_matrix,
    )


def select_programs_greedy(
    programs: list[str],
    budget: int,
    distance_matrix: Optional[np.ndarray] = None,
    delta: Optional[Callable] = None,
    show_progress: bool = True,
) -> SelectionResult:
    """Select diverse programs using greedy max-min selection.
    
    Alternative to hierarchical clustering. Iteratively selects the program
    that is furthest from all already-selected programs (maximin diversity).
    
    This is faster than hierarchical clustering for small budgets but may
    be less representative of the overall distribution.
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        distance_matrix: Pre-computed distance matrix (optional)
        delta: Optional custom cost function for edit distance
        show_progress: Whether to show progress bars
        
    Returns:
        SelectionResult with selected programs and metadata
    """
    n = len(programs)
    
    if budget >= n:
        return SelectionResult(
            selected_programs=programs.copy(),
            selected_indices=list(range(n)),
            cluster_labels=np.arange(n),
            cluster_sizes={i: 1 for i in range(n)},
            distance_matrix=distance_matrix,
        )
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        if show_progress:
            print(f"Computing {n}x{n} distance matrix ({n*(n-1)//2:,} pairs)...")
        distance_matrix = compute_distance_matrix(programs, delta, show_progress)
    
    # Greedy selection
    selected_indices = []
    
    # Start with program that has maximum total distance to others (most unique)
    total_distances = distance_matrix.sum(axis=1)
    first_idx = np.argmax(total_distances)
    selected_indices.append(first_idx)
    
    # Track minimum distance to selected set for each program
    min_dist_to_selected = distance_matrix[first_idx].copy()
    
    iterator = range(budget - 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Greedy selection")
    
    for _ in iterator:
        # Select program with maximum minimum distance to selected set
        # (exclude already selected)
        min_dist_to_selected[selected_indices] = -np.inf
        next_idx = np.argmax(min_dist_to_selected)
        selected_indices.append(next_idx)
        
        # Update minimum distances
        new_distances = distance_matrix[next_idx]
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_distances)
    
    # Assign cluster labels based on nearest selected program
    cluster_labels = np.zeros(n, dtype=int)
    for i in range(n):
        distances_to_selected = distance_matrix[i, selected_indices]
        cluster_labels[i] = np.argmin(distances_to_selected) + 1  # 1-indexed
    
    # Count cluster sizes
    cluster_sizes = {}
    for cluster_id in range(1, budget + 1):
        cluster_sizes[cluster_id] = int((cluster_labels == cluster_id).sum())
    
    selected_programs = [programs[i] for i in selected_indices]
    
    return SelectionResult(
        selected_programs=selected_programs,
        selected_indices=selected_indices,
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
        distance_matrix=distance_matrix,
    )


def deduplicate_programs(
    programs: list[str],
    tolerance: float = 0.0,
    delta: Optional[Callable] = None,
) -> tuple[list[str], list[int]]:
    """Remove duplicate or near-duplicate programs.
    
    Programs with edit distance <= tolerance are considered duplicates.
    Keeps the first occurrence of each unique program.
    
    Args:
        programs: List of program strings
        tolerance: Maximum edit distance to consider as duplicate (0 = exact match)
        delta: Optional custom cost function
        
    Returns:
        Tuple of (unique_programs, original_indices)
    """
    if not programs:
        return [], []
    
    trees = [parse_to_tree(p) for p in programs]
    unique_indices = [0]  # First program is always unique
    
    for i in range(1, len(programs)):
        is_duplicate = False
        for j in unique_indices:
            dist = program_edit_distance(trees[i], trees[j], delta)
            if dist <= tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_indices.append(i)
    
    unique_programs = [programs[i] for i in unique_indices]
    return unique_programs, unique_indices


def select_diverse_programs(
    programs: list[str],
    budget: int,
    method: str = "hierarchical",
    deduplicate: bool = True,
    dedup_tolerance: float = 0.0,
    cache_path: Optional[str] = None,
    **kwargs,
) -> SelectionResult:
    """High-level API for diverse program selection.
    
    This is the main entry point for program budget selection. It:
    1. Optionally deduplicates exact/near-duplicates
    2. Computes or loads cached distance matrix
    3. Applies the specified selection method
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        method: Selection method ('hierarchical' or 'greedy')
        deduplicate: Whether to remove duplicates first
        dedup_tolerance: Tolerance for deduplication (0 = exact match only)
        cache_path: Path to cache/load distance matrix (numpy .npy file)
        **kwargs: Additional arguments passed to selection method
        
    Returns:
        SelectionResult with selected programs and metadata
        
    Example:
        >>> programs = ["[0,50]head.y(1.6)", "[0,50]head.y(1.6)", "[0,30]lhand.x(0.3)", ...]
        >>> result = select_diverse_programs(programs, budget=100)
        >>> print(result.summary())
        Selected 100 programs from 500 total
        Number of clusters: 100
        Cluster sizes: min=1, max=15, mean=5.0
    """
    working_programs = programs
    index_mapping = list(range(len(programs)))
    
    # Step 1: Deduplicate if requested
    if deduplicate:
        working_programs, unique_indices = deduplicate_programs(
            programs, tolerance=dedup_tolerance
        )
        index_mapping = unique_indices
        
        if len(working_programs) < len(programs):
            n_removed = len(programs) - len(working_programs)
            print(f"Removed {n_removed} duplicate programs ({len(working_programs)} unique)")
    
    # Adjust budget if we have fewer programs than requested
    actual_budget = min(budget, len(working_programs))
    if actual_budget < budget:
        print(f"Adjusted budget from {budget} to {actual_budget} (fewer unique programs)")
    
    # Step 2: Load or compute distance matrix
    distance_matrix = None
    if cache_path:
        import os
        if os.path.exists(cache_path):
            print(f"Loading cached distance matrix from {cache_path}")
            distance_matrix = np.load(cache_path)
            if distance_matrix.shape[0] != len(working_programs):
                print(f"Cache size mismatch, recomputing...")
                distance_matrix = None
    
    # Step 3: Apply selection method
    if method == "hierarchical":
        result = select_programs_hierarchical(
            working_programs,
            actual_budget,
            distance_matrix=distance_matrix,
            **kwargs,
        )
    elif method == "greedy":
        result = select_programs_greedy(
            working_programs,
            actual_budget,
            distance_matrix=distance_matrix,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    # Cache distance matrix if path provided and not already cached
    if cache_path and result.distance_matrix is not None:
        import os
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, result.distance_matrix)
            print(f"Cached distance matrix to {cache_path}")
    
    # Map indices back to original if we deduplicated
    if deduplicate and len(index_mapping) < len(programs):
        original_indices = [index_mapping[i] for i in result.selected_indices]
        result = SelectionResult(
            selected_programs=result.selected_programs,
            selected_indices=original_indices,
            cluster_labels=result.cluster_labels,
            cluster_sizes=result.cluster_sizes,
            distance_matrix=result.distance_matrix,
        )
    
    return result
