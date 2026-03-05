"""Program edit distance using unordered tree edit distance (UTED).

Converts ExAct programs to tree representations and computes edit distances
for activity separability analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Union, Optional

from lark import Lark, Tree, Token


# Load grammar once at module level
_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
with open(_GRAMMAR_PATH, "r") as f:
    _GRAMMAR = f.read()
_PARSER = Lark(_GRAMMAR, start="start", parser="earley")


# Value tolerance for comparing numeric values
VALUE_TOLERANCE = 0.3


@dataclass
class ProgramTree:
    """Tree representation of a program for edit distance computation.
    
    Stores nodes and adjacency list in DFS order as required by edist.uted.
    
    Attributes:
        nodes: List of node labels (strings)
        adj: Adjacency list where adj[i] contains indices of children of node i
        program: Original program string
    """
    nodes: list[str]
    adj: list[list[int]]
    program: str
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __repr__(self) -> str:
        return f"ProgramTree({len(self.nodes)} nodes, program={self.program[:50]}...)"


def _normalize_value(value: float) -> str:
    """Normalize numeric value to a bucket for tolerance-aware comparison.
    
    Values within VALUE_TOLERANCE are mapped to the same bucket.
    This makes values like 0.3 and 0.5 (diff=0.2 < 0.3) compare as equal.
    
    Args:
        value: Numeric value from program
        
    Returns:
        Bucket label string (e.g., "v:0.3" for values in [0.15, 0.45))
    """
    # Round to nearest bucket center
    bucket = round(value / VALUE_TOLERANCE) * VALUE_TOLERANCE
    return f"v:{bucket:.1f}"


def _tree_to_nodes_adj(tree: Union[Tree, Token], nodes: list, adj: list, parent_idx: Optional[int] = None) -> int:
    """Convert Lark tree to nodes/adjacency list in DFS order.
    
    Args:
        tree: Lark Tree or Token
        nodes: List to append node labels to
        adj: List to append adjacency lists to
        parent_idx: Index of parent node (None for root)
        
    Returns:
        Index of this node in the nodes list
    """
    # Get current node index
    node_idx = len(nodes)
    
    if isinstance(tree, Token):
        # Terminal node - use token type and normalized value
        if tree.type in ("VALUE", "NUMBER"):
            label = _normalize_value(float(tree.value))
        elif tree.type in ("INT", "FRAME"):
            # Skip frame indices — we don't penalize different timings
            return -1
        else:
            # JOINT, AXIS — use value directly
            label = tree.value
    else:
        # Non-terminal - use rule name
        label = tree.data
    
    # Add node
    nodes.append(label)
    adj.append([])
    
    # Add to parent's children
    if parent_idx is not None:
        adj[parent_idx].append(node_idx)
    
    # Process children if Tree
    if isinstance(tree, Tree):
        for child in tree.children:
            child_idx = _tree_to_nodes_adj(child, nodes, adj, node_idx)
            # child_idx will be added to adj[node_idx] in recursive call
    
    return node_idx


def _lark_tree_to_program_tree(tree: Tree, program: str) -> ProgramTree:
    """Convert Lark parse tree to ProgramTree for edit distance.
    
    Simplifies the tree structure:
    - Removes interval values (INT tokens) to focus on structure
    - Normalizes numeric values for tolerance-aware comparison
    
    Args:
        tree: Lark parse tree
        program: Original program string
        
    Returns:
        ProgramTree with nodes and adjacency list
    """
    nodes = []
    adj = []
    
    # Build simplified tree without intervals
    _build_simplified_tree(tree, nodes, adj, None)
    
    return ProgramTree(nodes=nodes, adj=adj, program=program)


def _build_simplified_tree(tree: Union[Tree, Token], nodes: list, adj: list, parent_idx: Optional[int], in_sensor: bool = False) -> Optional[int]:
    """Build simplified tree structure, skipping interval values.
    
    Args:
        tree: Lark Tree or Token
        nodes: List to append node labels to
        adj: List to append adjacency lists to
        parent_idx: Index of parent node
        in_sensor: Whether we're inside a sensor node (VALUE should be kept)
    
    Returns:
        Index of created node, or None if skipped
    """
    if isinstance(tree, Token):
        # Skip frame indices (NUMBER tokens outside sensors, or FRAME/INT)
        if tree.type in ("FRAME", "INT"):
            return None
        if tree.type == "NUMBER" and not in_sensor:
            return None
        
        # Create node for token
        node_idx = len(nodes)
        
        if tree.type in ("VALUE", "NUMBER"):
            # This is a sensor value - normalize it
            label = _normalize_value(float(tree.value))
        else:
            label = tree.value
            
        nodes.append(label)
        adj.append([])
        
        if parent_idx is not None:
            adj[parent_idx].append(node_idx)
            
        return node_idx
    
    # Tree node
    node_idx = len(nodes)
    nodes.append(tree.data)
    adj.append([])
    
    if parent_idx is not None:
        adj[parent_idx].append(node_idx)
    
    # Process children - set in_sensor=True when entering a sensor node
    child_in_sensor = in_sensor or tree.data == "sensor"
    for child in tree.children:
        _build_simplified_tree(child, nodes, adj, node_idx, child_in_sensor)
    
    return node_idx


def parse_to_tree(program: str) -> ProgramTree:
    """Parse a program string into a tree representation.
    
    Args:
        program: ExAct program string (e.g., "[0,50]lhand.x(0.3)*acc;")
        
    Returns:
        ProgramTree for edit distance computation
        
    Raises:
        lark.exceptions.LarkError: If program cannot be parsed
    """
    tree = _PARSER.parse(program)
    return _lark_tree_to_program_tree(tree, program)


def program_edit_distance(
    program1: Union[str, ProgramTree],
    program2: Union[str, ProgramTree],
    delta: Optional[Callable] = None,
) -> float:
    """Compute unordered tree edit distance between two programs.
    
    Uses the constrained UTED algorithm from edist library.
    
    Args:
        program1: First program (string or ProgramTree)
        program2: Second program (string or ProgramTree)
        delta: Optional custom cost function. If None, uses unit costs.
               Signature: delta(node1, node2) -> float
               delta(node, None) = deletion cost
               delta(None, node) = insertion cost
               
    Returns:
        Edit distance (float >= 0)
    """
    from edist.uted import uted
    
    # Parse if needed
    if isinstance(program1, str):
        program1 = parse_to_tree(program1)
    if isinstance(program2, str):
        program2 = parse_to_tree(program2)
    
    # Compute UTED
    return uted(
        program1.nodes, program1.adj,
        program2.nodes, program2.adj,
        delta=delta
    )


def min_distance_to_model(
    query_program: Union[str, ProgramTree],
    model_programs: list[Union[str, ProgramTree]],
    delta: Optional[Callable] = None,
) -> tuple[float, int]:
    """Compute minimum edit distance from query to any program in model.
    
    Args:
        query_program: Query program
        model_programs: List of programs in the model
        delta: Optional custom cost function
        
    Returns:
        Tuple of (min_distance, index of closest program)
    """
    if isinstance(query_program, str):
        query_program = parse_to_tree(query_program)
    
    min_dist = float('inf')
    min_idx = -1
    
    for i, model_prog in enumerate(model_programs):
        dist = program_edit_distance(query_program, model_prog, delta)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_dist, min_idx


def batch_min_distances(
    query_programs: list[Union[str, ProgramTree]],
    model_programs: list[Union[str, ProgramTree]],
    delta: Optional[Callable] = None,
) -> list[float]:
    """Compute minimum distances for a batch of queries.
    
    Args:
        query_programs: List of query programs
        model_programs: List of programs in the model
        delta: Optional custom cost function
        
    Returns:
        List of minimum distances, one per query
    """
    # Pre-parse model programs
    model_trees = [
        parse_to_tree(p) if isinstance(p, str) else p
        for p in model_programs
    ]
    
    distances = []
    for query in query_programs:
        dist, _ = min_distance_to_model(query, model_trees, delta)
        distances.append(dist)
    
    return distances


def _compute_row_distances(
    query_activity_idx: int,
    query_activity: str,
    query_programs_data: list[tuple[list[str], list[list[int]], str]],
    model_activities: list[str],
    model_programs_data: dict[str, list[tuple[list[str], list[list[int]], str]]],
    n_activities: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute one row of the distance matrix (for parallel execution).
    
    Args:
        query_activity_idx: Index of query activity in matrix
        query_activity: Name of query activity
        query_programs_data: List of (nodes, adj, program) tuples for queries
        model_activities: List of all activity names
        model_programs_data: Dict mapping activity -> list of (nodes, adj, program)
        n_activities: Number of activities
        
    Returns:
        Tuple of (row_idx, mean_distances, std_distances)
    """
    from edist.uted import uted
    
    row_means = np.zeros(n_activities)
    row_stds = np.zeros(n_activities)
    
    # Reconstruct ProgramTree objects
    query_trees = [
        ProgramTree(nodes=nodes, adj=adj, program=prog)
        for nodes, adj, prog in query_programs_data
    ]
    
    for j, model_activity in enumerate(model_activities):
        if model_activity not in model_programs_data:
            continue
            
        model_data = model_programs_data[model_activity]
        model_trees = [
            ProgramTree(nodes=nodes, adj=adj, program=prog)
            for nodes, adj, prog in model_data
        ]
        
        # Compute min distance for each query
        distances = []
        for query in query_trees:
            min_dist = float('inf')
            for model in model_trees:
                dist = uted(
                    query.nodes, query.adj,
                    model.nodes, model.adj,
                    delta=None
                )
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        
        row_means[j] = np.mean(distances)
        row_stds[j] = np.std(distances)
    
    return query_activity_idx, row_means, row_stds


class ProgramDistanceMatrix:
    """Compute and store separability matrix using program edit distances.
    
    For each activity pair (i, j), computes:
        M[i,j] = mean(min_dist(q, model_j) for q in test_programs_i)
    
    A good separability shows low diagonal (same-activity distance)
    and high off-diagonal (cross-activity distance).
    """
    
    def __init__(self, activity_names: list[str]):
        """Initialize with activity names.
        
        Args:
            activity_names: List of activity names (defines matrix order)
        """
        self.activity_names = list(activity_names)
        self.n_activities = len(activity_names)
        self._activity_to_idx = {name: i for i, name in enumerate(activity_names)}
        
        # Model programs per activity (from training)
        self.model_programs: dict[str, list[ProgramTree]] = {}
        
        # Test programs per activity
        self.test_programs: dict[str, list[ProgramTree]] = {}
        
        # Results
        self.matrix: np.ndarray | None = None
        self.matrix_std: np.ndarray | None = None
    
    def set_model_programs(self, activity: str, programs: list[Union[str, ProgramTree]]):
        """Set model programs for an activity (from training data).
        
        Args:
            activity: Activity name
            programs: List of programs for this activity's model
        """
        self.model_programs[activity] = [
            parse_to_tree(p) if isinstance(p, str) else p
            for p in programs
        ]
    
    def add_test_program(self, activity: str, program: Union[str, ProgramTree]):
        """Add a test program for an activity.
        
        Args:
            activity: Ground-truth activity of the program
            program: Program to add
        """
        if activity not in self.test_programs:
            self.test_programs[activity] = []
        
        tree = parse_to_tree(program) if isinstance(program, str) else program
        self.test_programs[activity].append(tree)
    
    def compute_matrix(self, delta: Optional[Callable] = None, verbose: bool = True, num_workers: int = 1) -> np.ndarray:
        """Compute the separability matrix.
        
        Args:
            delta: Optional custom cost function
            verbose: Whether to show progress
            num_workers: Number of parallel workers (default: 1 = sequential)
            
        Returns:
            Matrix M where M[i,j] = mean min-distance from activity i tests to activity j model
        """
        import numpy as np
        from tqdm import tqdm
        
        if num_workers > 1:
            return self._compute_matrix_parallel(num_workers, verbose)
        
        matrix = np.zeros((self.n_activities, self.n_activities))
        matrix_std = np.zeros((self.n_activities, self.n_activities))
        
        iterator = self.activity_names
        if verbose:
            iterator = tqdm(iterator, desc="Computing distance matrix")
        
        for i, query_activity in enumerate(iterator):
            if query_activity not in self.test_programs:
                continue
                
            query_programs = self.test_programs[query_activity]
            
            for j, model_activity in enumerate(self.activity_names):
                if model_activity not in self.model_programs:
                    continue
                
                model_progs = self.model_programs[model_activity]
                
                # Compute min distance for each query
                distances = batch_min_distances(query_programs, model_progs, delta)
                
                matrix[i, j] = np.mean(distances)
                matrix_std[i, j] = np.std(distances)
        
        self.matrix = matrix
        self.matrix_std = matrix_std
        
        return matrix
    
    def _compute_matrix_parallel(self, num_workers: int, verbose: bool = True) -> np.ndarray:
        """Compute the separability matrix using parallel workers.
        
        Args:
            num_workers: Number of parallel workers
            verbose: Whether to show progress
            
        Returns:
            Matrix M where M[i,j] = mean min-distance from activity i tests to activity j model
        """
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        matrix = np.zeros((self.n_activities, self.n_activities))
        matrix_std = np.zeros((self.n_activities, self.n_activities))
        
        # Convert ProgramTree objects to serializable tuples
        model_programs_data = {}
        for activity, trees in self.model_programs.items():
            model_programs_data[activity] = [
                (t.nodes, t.adj, t.program) for t in trees
            ]
        
        # Prepare tasks - one per query activity
        tasks = []
        for i, query_activity in enumerate(self.activity_names):
            if query_activity not in self.test_programs:
                continue
            
            query_data = [
                (t.nodes, t.adj, t.program)
                for t in self.test_programs[query_activity]
            ]
            tasks.append((i, query_activity, query_data))
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for i, query_activity, query_data in tasks:
                future = executor.submit(
                    _compute_row_distances,
                    i,
                    query_activity,
                    query_data,
                    self.activity_names,
                    model_programs_data,
                    self.n_activities,
                )
                futures[future] = query_activity
            
            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(iterator, total=len(futures), desc="Computing distance matrix (parallel)")
            
            for future in iterator:
                row_idx, row_means, row_stds = future.result()
                matrix[row_idx, :] = row_means
                matrix_std[row_idx, :] = row_stds
        
        self.matrix = matrix
        self.matrix_std = matrix_std
        
        return matrix
    
    def get_separability_metrics(self) -> dict:
        """Compute separability metrics from the matrix.
        
        Returns:
            Dict with:
            - diagonal_mean: Mean of diagonal (same-activity distance)
            - off_diagonal_mean: Mean of off-diagonal (cross-activity distance)
            - separation: off_diagonal_mean - diagonal_mean (higher = better)
            - per_activity: Per-activity separation scores
        """
        if self.matrix is None:
            raise ValueError("Must call compute_matrix() first")
        
        import numpy as np
        
        n = self.n_activities
        diag = np.diag(self.matrix)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = self.matrix[off_diag_mask]
        
        # Per-activity: compare diagonal to row mean (excluding diagonal)
        per_activity = {}
        for i, activity in enumerate(self.activity_names):
            row = self.matrix[i, :]
            row_off_diag = np.concatenate([row[:i], row[i+1:]])
            per_activity[activity] = {
                "same_activity_dist": float(diag[i]),
                "cross_activity_dist": float(np.mean(row_off_diag)),
                "separation": float(np.mean(row_off_diag) - diag[i]),
            }
        
        return {
            "diagonal_mean": float(np.mean(diag)),
            "off_diagonal_mean": float(np.mean(off_diag)),
            "separation": float(np.mean(off_diag) - np.mean(diag)),
            "per_activity": per_activity,
        }
    
    def compute_score_matrix(
        self,
        method: str = "mean-sigmoid",
        verbose: bool = True,
        num_workers: int = 1,
    ) -> np.ndarray:
        """Compute a score matrix using sigmoid-based similarity scoring.

        For each test program q from activity i scored against model j:
        - mean-sigmoid: score = (1/N) * sum_k sigma(-edist(model_k, q))
        - min-sigmoid:  score = sigma(-min_k edist(model_k, q))

        Higher score = test program is more similar to model activity.

        Args:
            method: "mean-sigmoid" or "min-sigmoid"
            verbose: Whether to show progress
            num_workers: Number of parallel workers (unused for now, sequential)

        Returns:
            Matrix S where S[i,j] = mean score of activity i test programs against model j
        """
        from tqdm import tqdm

        if method not in ("mean-sigmoid", "min-sigmoid"):
            raise ValueError(f"Unknown method: {method}. Use 'mean-sigmoid' or 'min-sigmoid'.")

        score_matrix = np.zeros((self.n_activities, self.n_activities))

        iterator = self.activity_names
        if verbose:
            iterator = tqdm(iterator, desc=f"Computing score matrix ({method})")

        for i, query_activity in enumerate(iterator):
            if query_activity not in self.test_programs:
                continue

            query_programs = self.test_programs[query_activity]

            for j, model_activity in enumerate(self.activity_names):
                if model_activity not in self.model_programs:
                    continue

                model_progs = self.model_programs[model_activity]
                scores = _compute_sigmoid_scores(query_programs, model_progs, method)
                score_matrix[i, j] = np.mean(scores)

        return score_matrix

    def compute_score_matrix_per_program(
        self,
        method: str = "mean-sigmoid",
        verbose: bool = True,
    ) -> dict[str, dict[str, list[float]]]:
        """Compute per-program scores for AUC computation.

        Returns:
            Dict[query_activity][model_activity] -> list of scores (one per test program)
        """
        from tqdm import tqdm

        if method not in ("mean-sigmoid", "min-sigmoid"):
            raise ValueError(f"Unknown method: {method}. Use 'mean-sigmoid' or 'min-sigmoid'.")

        per_program_scores: dict[str, dict[str, list[float]]] = {}

        iterator = self.activity_names
        if verbose:
            iterator = tqdm(iterator, desc=f"Computing per-program scores ({method})")

        for query_activity in iterator:
            if query_activity not in self.test_programs:
                continue

            query_programs = self.test_programs[query_activity]
            per_program_scores[query_activity] = {}

            for model_activity in self.activity_names:
                if model_activity not in self.model_programs:
                    continue

                model_progs = self.model_programs[model_activity]
                scores = _compute_sigmoid_scores(query_programs, model_progs, method)
                per_program_scores[query_activity][model_activity] = scores

        return per_program_scores

    def compute_auc_matrix(
        self,
        method: str = "mean-sigmoid",
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Compute AUC matrix using sigmoid scoring.

        Follows the same convention as the NF anomaly detection AUC matrix:

        AUC[i,j] (off-diagonal): AUROC of model i separating its own activity i
                  (positive) from activity j (negative).
                  = P(score_i(x_i) > score_i(x_j))
        AUC[i,i] (diagonal): one-vs-rest AUROC for model i — how well model i
                  separates activity i from all other activities pooled.

        Where score_i(x) is the sigmoid-based similarity score of program x
        evaluated against model i's programs.

        Args:
            method: "mean-sigmoid" or "min-sigmoid"
            verbose: Whether to show progress

        Returns:
            Tuple of:
            - auc_matrix: N×N AUROC matrix (row=model/target, col=query/negative)
            - score_matrix: N×N matrix of mean scores S[i,j]
              where S[i,j] = mean score of activity i test programs against model j
            - metrics: Dict with per-model AUC, mean AUC, etc.
        """
        from sklearn.metrics import roc_auc_score

        # Get per-program scores
        # per_program_scores[query_activity][model_activity] -> list of scores
        per_program_scores = self.compute_score_matrix_per_program(method, verbose)

        # Compute score matrix (mean over test programs)
        # score_matrix[i,j] = mean score of activity i test programs against model j
        score_matrix = np.zeros((self.n_activities, self.n_activities))
        for i, qa in enumerate(self.activity_names):
            for j, ma in enumerate(self.activity_names):
                scores = per_program_scores.get(qa, {}).get(ma, [])
                score_matrix[i, j] = np.mean(scores) if scores else 0.0

        # Compute pairwise AUC matrix
        # Convention: auc_matrix[i, j] = AUROC of model i separating activity i from activity j
        # Row = model (target), Column = query (negative class)
        auc_matrix = np.full((self.n_activities, self.n_activities), np.nan)

        for i, model_activity in enumerate(self.activity_names):
            # Positive: test programs from model_activity scored by this model
            pos_scores = per_program_scores.get(model_activity, {}).get(model_activity, [])
            if not pos_scores:
                continue

            # Off-diagonal: pairwise AUROC against each other activity
            for j, query_activity in enumerate(self.activity_names):
                if i == j:
                    continue  # diagonal handled below

                # Negative: test programs from query_activity scored by model i
                neg_scores = per_program_scores.get(query_activity, {}).get(model_activity, [])
                if not neg_scores:
                    continue

                # AUC: can model i tell apart its own activity from activity j?
                y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
                y_score = list(pos_scores) + list(neg_scores)

                try:
                    auc_matrix[i, j] = roc_auc_score(y_true, y_score)
                except ValueError:
                    pass

            # Diagonal: one-vs-rest AUROC for model i
            all_neg_scores = []
            for j, query_activity in enumerate(self.activity_names):
                if i == j:
                    continue
                neg = per_program_scores.get(query_activity, {}).get(model_activity, [])
                all_neg_scores.extend(neg)

            if all_neg_scores:
                y_true = [1] * len(pos_scores) + [0] * len(all_neg_scores)
                y_score = list(pos_scores) + list(all_neg_scores)
                try:
                    auc_matrix[i, i] = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    pass

        # Per-model aggregate AUC = diagonal values
        per_model_auc = {}
        for i, model_activity in enumerate(self.activity_names):
            per_model_auc[model_activity] = float(auc_matrix[i, i])

        # Summary metrics
        valid_aucs = [v for v in per_model_auc.values() if not np.isnan(v)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        # Off-diagonal AUC mean
        off_diag_mask = ~np.eye(self.n_activities, dtype=bool)
        valid_pairwise = auc_matrix[off_diag_mask]
        valid_pairwise = valid_pairwise[~np.isnan(valid_pairwise)]
        mean_pairwise_auc = float(np.mean(valid_pairwise)) if len(valid_pairwise) > 0 else float("nan")

        metrics = {
            "method": method,
            "mean_auc": mean_auc,
            "mean_pairwise_auc": mean_pairwise_auc,
            "per_model_auc": per_model_auc,
        }

        return auc_matrix, score_matrix, metrics

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        import numpy as np
        
        return {
            "activity_names": self.activity_names,
            "matrix": self.matrix.tolist() if self.matrix is not None else None,
            "matrix_std": self.matrix_std.tolist() if self.matrix_std is not None else None,
            "metrics": self.get_separability_metrics() if self.matrix is not None else None,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _compute_sigmoid_scores(
    query_programs: list[ProgramTree],
    model_programs: list[ProgramTree],
    method: str,
) -> list[float]:
    """Compute sigmoid-based similarity scores for each query against a model.

    Args:
        query_programs: Test programs to score
        model_programs: Model programs (from training)
        method: "mean-sigmoid" or "min-sigmoid"

    Returns:
        List of scores, one per query program (higher = more similar to model)
    """
    from edist.uted import uted

    scores = []
    for query in query_programs:
        # Compute distances to all model programs
        dists = np.array([
            uted(
                query.nodes, query.adj,
                model.nodes, model.adj,
                delta=None,
            )
            for model in model_programs
        ])

        if method == "mean-sigmoid":
            # mean over i of sigma(-edist(model_i, query))
            score = float(np.mean(_sigmoid(-dists)))
        elif method == "min-sigmoid":
            # sigma(-min_i edist(model_i, query))
            score = float(_sigmoid(np.array([-dists.min()]))[0])
        else:
            raise ValueError(f"Unknown method: {method}")

        scores.append(score)

    return scores


# Need numpy for matrix operations
import numpy as np
