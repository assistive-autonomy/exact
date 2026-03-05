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
        
        if len(cluster_indices) == 0:
            # Empty cluster (can happen with degenerate cases)
            continue
        elif len(cluster_indices) == 1:
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


def extract_program_features(program: str) -> dict:
    """Extract features from a program string for TF-IDF-like vectorization.
    
    Extracts:
    - Predicates as tokens: joint.axis(value_bucket) → "head_y_1.5"
    - Segment count
    - Program length (character count)
    - Temporal span coverage
    
    Args:
        program: Program string like "[0,100]head.y(1.5)*rhand.x(0.3);[100,200]lknee.z(0.8)"
        
    Returns:
        Dict with 'tokens' (list of predicate tokens), 'num_segments', 'length', etc.
    """
    import re
    
    features = {
        'tokens': [],
        'num_segments': 0,
        'length': len(program),
        'total_predicates': 0,
        'joints': set(),
        'axes': set(),
    }
    
    # Parse segments
    segments = program.split(';')
    features['num_segments'] = len([s for s in segments if s.strip()])
    
    # Extract predicates from each segment
    predicate_pattern = re.compile(r'([a-z_]+)\.([xyz])\(([-\d.]+)\)')
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        # Find all predicates in this segment
        for match in predicate_pattern.finditer(segment):
            joint = match.group(1)
            axis = match.group(2)
            value = float(match.group(3))
            
            # Bucket the value (0.3 granularity, like edit_distance.py)
            value_bucket = round(value / 0.3) * 0.3
            
            # Create token: joint_axis_value
            token = f"{joint}_{axis}_{value_bucket:.1f}"
            features['tokens'].append(token)
            features['joints'].add(joint)
            features['axes'].add(axis)
            features['total_predicates'] += 1
    
    return features


def select_programs_tfidf(
    programs: list[str],
    budget: int,
    show_progress: bool = True,
    length_weight: float = 0.1,
    segment_weight: float = 0.2,
) -> SelectionResult:
    """Select diverse programs using TF-IDF-like feature vectorization.
    
    This method is much faster than tree edit distance (O(n·d) vs O(n²·k²))
    and considers:
    - Predicate diversity (joint.axis combinations with bucketed values)
    - Program structure (segment count)
    - Program length
    
    The approach:
    1. Extract tokens from each program (predicates like "head_y_1.5")
    2. Build TF-IDF matrix over the token vocabulary
    3. Add structural features (segment count, length) as extra dimensions
    4. Use k-means++ clustering to select diverse programs as cluster centroids
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        show_progress: Whether to show progress
        length_weight: Weight for program length feature (0-1)
        segment_weight: Weight for segment count feature (0-1)
        
    Returns:
        SelectionResult with selected programs and metadata
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix
    
    n = len(programs)
    
    if budget >= n:
        return SelectionResult(
            selected_programs=programs.copy(),
            selected_indices=list(range(n)),
            cluster_labels=np.arange(n),
            cluster_sizes={i: 1 for i in range(n)},
            distance_matrix=None,
        )
    
    if show_progress:
        print(f"Extracting features from {n} programs...")
    
    # Extract features from all programs
    all_features = [extract_program_features(p) for p in programs]
    
    # Build token strings for TF-IDF (space-separated predicates)
    token_strings = [' '.join(f['tokens']) for f in all_features]
    
    # Handle edge case: all programs have no predicates
    if all(not ts.strip() for ts in token_strings):
        # Fall back to random selection if no predicates
        import random
        rng = random.Random(42)
        indices = rng.sample(range(n), budget)
        return SelectionResult(
            selected_programs=[programs[i] for i in indices],
            selected_indices=indices,
            cluster_labels=np.zeros(n, dtype=int),
            cluster_sizes={0: n},
            distance_matrix=None,
        )
    
    # Build TF-IDF matrix
    if show_progress:
        print("Building TF-IDF matrix...")
    
    vectorizer = TfidfVectorizer(
        token_pattern=r'[^\s]+',  # Each space-separated token
        lowercase=False,
        norm='l2',
    )
    tfidf_matrix = vectorizer.fit_transform(token_strings)
    
    # Extract structural features
    lengths = np.array([f['length'] for f in all_features]).reshape(-1, 1)
    segments = np.array([f['num_segments'] for f in all_features]).reshape(-1, 1)
    predicates = np.array([f['total_predicates'] for f in all_features]).reshape(-1, 1)
    n_joints = np.array([len(f['joints']) for f in all_features]).reshape(-1, 1)
    
    # Normalize structural features
    scaler = StandardScaler()
    structural = np.hstack([lengths, segments, predicates, n_joints])
    structural_scaled = scaler.fit_transform(structural)
    
    # Weight structural features
    structural_weighted = structural_scaled * np.array([
        length_weight, segment_weight, 0.1, 0.1  # weights for each feature
    ])
    
    # Combine TF-IDF with structural features
    combined_matrix = hstack([
        tfidf_matrix,
        csr_matrix(structural_weighted)
    ])
    
    if show_progress:
        print(f"Feature matrix shape: {combined_matrix.shape}")
        print(f"Running k-means clustering with k={budget}...")
    
    # Use k-means++ to find diverse cluster centers
    kmeans = KMeans(
        n_clusters=budget,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    cluster_labels = kmeans.fit_predict(combined_matrix)
    
    # Find medoid (closest to centroid) for each cluster
    selected_indices = []
    cluster_sizes = {}
    
    # Convert to dense for distance computation
    combined_dense = combined_matrix.toarray()
    
    for cluster_id in range(budget):
        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_sizes[cluster_id + 1] = len(cluster_indices)
        
        if len(cluster_indices) == 0:
            continue
        elif len(cluster_indices) == 1:
            selected_indices.append(cluster_indices[0])
        else:
            # Find point closest to cluster centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_points = combined_dense[cluster_indices]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            medoid_local_idx = np.argmin(distances)
            selected_indices.append(cluster_indices[medoid_local_idx])
    
    # Convert cluster labels to 1-indexed
    cluster_labels_1indexed = cluster_labels + 1
    
    selected_programs = [programs[i] for i in selected_indices]
    
    if show_progress:
        print(f"Selected {len(selected_programs)} diverse programs")
        # Print some stats about selected programs
        selected_features = [all_features[i] for i in selected_indices]
        avg_predicates = np.mean([f['total_predicates'] for f in selected_features])
        unique_joints = set()
        for f in selected_features:
            unique_joints.update(f['joints'])
        print(f"  Avg predicates per program: {avg_predicates:.1f}")
        print(f"  Unique joints covered: {len(unique_joints)}")
    
    return SelectionResult(
        selected_programs=selected_programs,
        selected_indices=selected_indices,
        cluster_labels=cluster_labels_1indexed,
        cluster_sizes=cluster_sizes,
        distance_matrix=None,
    )


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
        method: Selection method ('hierarchical', 'greedy', or 'tfidf')
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
    
    # Step 2: Load or compute distance matrix (only for hierarchical/greedy)
    distance_matrix = None
    if method in ("hierarchical", "greedy") and cache_path:
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
    elif method == "tfidf":
        result = select_programs_tfidf(
            working_programs,
            actual_budget,
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


def select_programs_tfidf(
    programs: list[str],
    budget: int,
    show_progress: bool = True,
) -> SelectionResult:
    """Select diverse programs using TF-IDF vectorization and greedy max-min selection.
    
    This is a fast alternative to tree edit distance based methods. It:
    1. Extracts rich features from each program (joints, axes, predicates, structure)
    2. Computes TF-IDF vectors capturing predicate diversity
    3. Uses cosine distance for fast similarity computation
    4. Applies greedy max-min selection for diversity
    
    Features extracted:
    - Joint-axis pairs (e.g., "lhand_x", "rknee_z")
    - Value buckets (quantized to 0.3 intervals)
    - Segment count (num_segs_1, num_segs_2, ...)
    - Predicates per segment ratios
    - Temporal span features (early, mid, late coverage)
    - Body part categories (arm, leg, torso, head, left/right)
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        show_progress: Whether to show progress bar
        
    Returns:
        SelectionResult with selected programs and metadata
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re
    
    n = len(programs)
    
    if budget >= n:
        return SelectionResult(
            selected_programs=programs.copy(),
            selected_indices=list(range(n)),
            cluster_labels=np.arange(n),
            cluster_sizes={i: 1 for i in range(n)},
            distance_matrix=None,
        )
    
    def program_to_features(program: str) -> str:
        """Convert program string to a bag of feature tokens."""
        features = []
        
        # Parse segments
        segments = program.strip().rstrip(';').split(';')
        num_segments = len(segments)
        features.append(f"num_segs_{min(num_segments, 10)}")
        
        total_predicates = 0
        covered_frames = set()
        all_joints = []
        all_axes = []
        
        for seg_idx, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            
            # Extract frame range [start,end]
            frame_match = re.match(r'\[(\d+),(\d+)\]', segment)
            if frame_match:
                start, end = int(frame_match.group(1)), int(frame_match.group(2))
                # Temporal position features (early, mid, late thirds of 1024 frames)
                if start < 341:
                    features.append("temporal_early")
                if start >= 341 and start < 682:
                    features.append("temporal_mid")
                if start >= 682:
                    features.append("temporal_late")
                # Segment duration feature
                duration = end - start
                if duration < 100:
                    features.append("duration_short")
                elif duration < 300:
                    features.append("duration_medium")
                else:
                    features.append("duration_long")
                # Track frame coverage
                for f in range(start, min(end, 1024)):
                    covered_frames.add(f)
            
            # Extract predicates (joint.axis(value))
            predicates = re.findall(r'(\w+)\.([xyz])\(([0-9.-]+)\)', segment)
            for joint, axis, value in predicates:
                # Joint-axis feature (most important for diversity)
                features.append(f"{joint}_{axis}")
                all_joints.append(joint)
                all_axes.append(axis)
                
                # Body part category features
                if joint.startswith('l'):
                    features.append("side_left")
                elif joint.startswith('r'):
                    features.append("side_right")
                
                if any(p in joint for p in ['hand', 'wrist', 'elbow']):
                    features.append("part_arm")
                elif any(p in joint for p in ['hip', 'knee', 'ankle', 'toe']):
                    features.append("part_leg")
                elif any(p in joint for p in ['shoulder', 'thorax', 'chest', 'spine', 'pelvis', 'torso']):
                    features.append("part_torso")
                elif any(p in joint for p in ['head', 'neck']):
                    features.append("part_head")
                
                # Value bucket (quantized)
                try:
                    v = float(value)
                    bucket = round(v / 0.3) * 0.3
                    features.append(f"val_{bucket:.1f}")
                    # Extremity features
                    if v < 0.3:
                        features.append("val_low")
                    elif v > 1.5:
                        features.append("val_high")
                except ValueError:
                    pass
                total_predicates += 1
            
            # Predicates per segment
            num_preds = len(predicates)
            features.append(f"preds_per_seg_{min(num_preds, 5)}")
        
        # Total predicates feature
        features.append(f"total_preds_{min(total_predicates, 15)}")
        
        # Joint diversity in program
        unique_joints = len(set(all_joints))
        features.append(f"unique_joints_{min(unique_joints, 10)}")
        
        # Axis coverage
        axis_set = set(all_axes)
        if 'x' in axis_set:
            features.append("uses_x")
        if 'y' in axis_set:
            features.append("uses_y")
        if 'z' in axis_set:
            features.append("uses_z")
        
        # Frame coverage ratio
        coverage = len(covered_frames) / 1024.0
        if coverage < 0.25:
            features.append("coverage_low")
        elif coverage < 0.5:
            features.append("coverage_med")
        elif coverage < 0.75:
            features.append("coverage_high")
        else:
            features.append("coverage_full")
        
        return " ".join(features)
    
    if show_progress:
        print(f"Extracting features from {n} programs...")
    
    # Convert all programs to feature strings
    feature_docs = [program_to_features(p) for p in programs]
    
    # Compute TF-IDF vectors
    if show_progress:
        print("Computing TF-IDF vectors...")
    
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'[a-zA-Z0-9_.-]+',
        min_df=1,
        max_df=0.95,  # Ignore features in >95% of programs
    )
    tfidf_matrix = vectorizer.fit_transform(feature_docs)
    
    if show_progress:
        print(f"Feature vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Greedy max-min selection using cosine distance
    selected_indices = []
    
    # Start with program that has highest L2 norm (most distinctive features)
    norms = np.asarray(tfidf_matrix.power(2).sum(axis=1)).flatten()
    first_idx = int(np.argmax(norms))
    selected_indices.append(first_idx)
    
    # Track minimum distance to selected set for each program
    # Using cosine distance: 1 - cosine_similarity
    selected_vectors = tfidf_matrix[first_idx]
    min_dist_to_selected = 1 - np.asarray(
        tfidf_matrix.dot(selected_vectors.T).todense()
    ).flatten()
    
    iterator = range(budget - 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Selecting diverse programs")
    
    for _ in iterator:
        # Mask already selected
        min_dist_to_selected[selected_indices] = -np.inf
        
        # Select program with maximum minimum distance
        next_idx = int(np.argmax(min_dist_to_selected))
        selected_indices.append(next_idx)
        
        # Update minimum distances with new selection
        new_vec = tfidf_matrix[next_idx]
        new_distances = 1 - np.asarray(
            tfidf_matrix.dot(new_vec.T).todense()
        ).flatten()
        min_dist_to_selected = np.minimum(min_dist_to_selected, new_distances)
    
    # Assign cluster labels based on nearest selected program
    if show_progress:
        print("Assigning cluster labels...")
    
    selected_matrix = tfidf_matrix[selected_indices]
    similarities = tfidf_matrix.dot(selected_matrix.T).toarray()
    cluster_labels = np.argmax(similarities, axis=1) + 1  # 1-indexed
    
    # Count cluster sizes
    cluster_sizes = {}
    for cluster_id in range(1, budget + 1):
        cluster_sizes[cluster_id] = int((cluster_labels == cluster_id).sum())
    
    selected_programs = [programs[i] for i in selected_indices]
    
    if show_progress:
        print(f"Selected {len(selected_programs)} diverse programs")
    
    return SelectionResult(
        selected_programs=selected_programs,
        selected_indices=selected_indices,
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
        distance_matrix=None,  # Not computed for TF-IDF method
    )