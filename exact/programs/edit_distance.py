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


# Value tolerance for comparing numeric values (kept for backward compat)
VALUE_TOLERANCE = 0.3

#: Width of each value quantisation bucket.  Values are binned into
#: intervals of this width: [0, BIN_WIDTH), [BIN_WIDTH, 2×BIN_WIDTH), …
#: so that movement-magnitude differences contribute to edit distance.
VALUE_BIN_WIDTH = 0.4
#: Maximum bucket index (values ≥ BIN_WIDTH×NUM are clamped).
VALUE_NUM_BINS = 5


# =============================================================================
# Body-region groupings for weighted edit distance
# =============================================================================

JOINT_GROUPS: dict[str, list[str]] = {
    "head":      ["head", "neck"],
    "torso":     ["pelvis", "torso", "spine", "chest"],
    "left_arm":  ["lthorax", "lshoulder", "lelbow", "lwrist", "lhand"],
    "right_arm": ["rthorax", "rshoulder", "relbow", "rwrist", "rhand"],
    "left_leg":  ["lhip", "lknee", "lankle", "ltoe"],
    "right_leg": ["rhip", "rknee", "rankle", "rtoe"],
}

# Reverse mapping: joint name -> group name
JOINT_TO_GROUP: dict[str, str] = {
    joint: group for group, joints in JOINT_GROUPS.items() for joint in joints
}

ALL_JOINTS: set[str] = set(JOINT_TO_GROUP.keys())
ALL_AXES: set[str] = {"x", "y", "z"}
STRUCTURAL_NODES: set[str] = {"start", "motion", "sensor"}

_MIRROR_GROUPS: dict[str, str] = {
    "left_arm": "right_arm", "right_arm": "left_arm",
    "left_leg": "right_leg", "right_leg": "left_leg",
}

# Adjacent body region pairs (upper ↔ lower on the same side)
_ADJACENT_GROUPS: set[frozenset[str]] = {
    frozenset({"left_arm", "left_leg"}),
    frozenset({"right_arm", "right_leg"}),
    frozenset({"torso", "left_arm"}), frozenset({"torso", "right_arm"}),
    frozenset({"torso", "left_leg"}), frozenset({"torso", "right_leg"}),
    frozenset({"head", "torso"}),
}


def _is_value_label(label: str) -> bool:
    """Check if a node label is a value/sign label."""
    if label.startswith("v:") or label in ("pos", "neg"):
        return True
    # New bucket format: "b0".."b4" (positive) or "n0".."n4" (negative)
    return len(label) == 2 and label[0] in ("b", "n") and label[1].isdigit()


def _get_value_from_label(label: str) -> float:
    """Extract numeric value from a value bucket label."""
    return float(label[2:])


def _parse_composite_label(label: str) -> tuple[str, str, str | None] | None:
    """Parse a composite sensor label ``"joint.axis:value"``.

    Returns ``(joint, axis, value_bucket)`` or ``None`` if *label* is
    not a composite sensor label.
    """
    if "." not in label:
        return None
    try:
        joint_axis, *rest = label.split(":", 1)
        joint, axis = joint_axis.split(".", 1)
    except ValueError:
        return None
    if joint not in ALL_JOINTS or axis not in ALL_AXES:
        return None
    value = rest[0] if rest else None
    return joint, axis, value


# -- Cost constants -----------------------------------------------------------
# Costs are calibrated so that:
#
#   1. *Correct joints and values get strong reward* — matching or nearby
#      sensors are nearly free (0.0–0.2), so programs from the same activity
#      produce small total distances (≈0–3).
#   2. *Completely wrong body parts are severely penalised* — COST_DISTANT
#      (4.0 per sensor pair) drives cross-activity distances to 8–20+.
#   3. The biased sigmoid ``σ(bias - k·d)`` maps d≈0 → score≈1.0 and
#      d≈8+ → score≈0, giving the full [0, 1] range in the AUC matrix.
#
# Hierarchy (low → high cost):
#   wrong sign < same region < axis < mirror < adjacent < indel < distant
#
# UTED optimality: 2×SENSOR_INDEL (5.2) > max substitution
# (DISTANT + AXIS + SIGN + 4×VALUE_STEP = 5.05) so the solver always
# substitutes rather than bypassing body-region costs via delete+insert.

#: Substitution: same joint & axis, wrong sign (best partial credit)
COST_SIGN_MISMATCH = 0.1
#: Graduated cost per value-bucket step (additive with sign mismatch).
#: Bucket 0 vs bucket 2 costs 2 × COST_VALUE_STEP = 0.30.
COST_VALUE_STEP = 0.15
#: Substitution: joint in same body region (e.g. lhand ↔ lwrist)
COST_SAME_REGION = 0.2
#: Substitution: different axis on same/similar joint
COST_AXIS = 0.35
#: Substitution: mirror joint (e.g. lhand ↔ rhand) — correct part, wrong side
COST_MIRROR = 0.5
#: Substitution: adjacent body region (e.g. lhand ↔ lhip, torso ↔ larm)
COST_ADJACENT = 1.5
#: Insertion / deletion of a composite sensor node
COST_SENSOR_INDEL = 2.6
#: Substitution: distant body region (e.g. lhand ↔ rankle) — **most** expensive
COST_DISTANT = 4.0
#: Insertion / deletion of a motion segment node (segment count matters slightly)
COST_MOTION_INDEL = 0.15
#: Insertion / deletion of the start node (structural, near-free)
COST_START_INDEL = 0.01
#: Kept for backward compatibility — maps to :data:`COST_START_INDEL`.
COST_STRUCTURAL_INDEL = COST_START_INDEL

#: Sigmoid bias.  ``σ(bias - k·d)`` with ``bias > 0`` means a perfect
#: match (d=0) scores ``σ(bias) ≈ 1.0``, giving the full [0, 1] range.
#: Calibrated for **raw** collapsed-tree distances in [0, ~8].
SIGMOID_BIAS = 4.0
#: Sigmoid temperature.  Controls decay speed with distance.
#: With bias=4, k=1:  d=0 → 0.98, d=2 → 0.88, d=4 → 0.50, d=6 → 0.12.
SIGMOID_TEMPERATURE = 1.0

def weighted_delta(a: Optional[str], b: Optional[str]) -> float:
    """Body-region-aware cost function for tree edit distance.

    Designed for *collapsed* program trees where each sensor is a single
    composite node ``"joint.axis:value"`` (e.g. ``"lhand.x:v0.3"``).
    Structural nodes (``start``, ``motion``) are nearly free so the
    distance is dominated by **which body parts** the programs use.

    The cost hierarchy reflects the hypothesis that activity similarity
    is driven by shared body-region involvement, with costs deliberately
    low so that overall tree structure (number/arrangement of sensors)
    dominates over individual joint identity:

    Composite sensor substitution (joint.axis:sign ↔ joint.axis:sign):
        Same label                   → 0.0
        Same joint+axis, wrong sign  → 0.1   (best partial credit)
        Same region joint            → 0.2   (+ axis/sign penalty)
        Different axis               → +0.35 (additive)
        Mirror joint (l↔r)           → 0.5   (+ axis/sign penalty)
        Adjacent region              → 1.5   (+ axis/sign penalty)
        Distant region               → 4.0   (+ axis/sign penalty)
        (axis mismatch adds 0.35, sign mismatch adds 0.1)

    Insertion / deletion:
        Composite sensor             → 2.5   (missing predicate)
        Motion segment               → 0.15  (segment count)
        Start node                   → 0.01  (structural, near-free)

    Falls back to per-token costs for non-collapsed (legacy) trees.
    """
    # ------------------------------------------------------------------
    # Deletion / insertion
    # ------------------------------------------------------------------
    if a is None or b is None:
        node = a if b is None else b
        # Composite sensor label?
        if _parse_composite_label(node) is not None:
            return COST_SENSOR_INDEL
        if node == "motion":
            return COST_MOTION_INDEL
        if node in STRUCTURAL_NODES:
            return COST_START_INDEL
        # Legacy individual tokens
        if node in ALL_JOINTS:
            return COST_SENSOR_INDEL
        if node in ALL_AXES:
            return 0.5
        if _is_value_label(node):
            return 0.25
        return 1.0

    # ------------------------------------------------------------------
    # Identical labels → zero cost
    # ------------------------------------------------------------------
    if a == b:
        return 0.0

    # ------------------------------------------------------------------
    # Both composite sensors (primary path for collapsed trees)
    # ------------------------------------------------------------------
    ca = _parse_composite_label(a)
    cb = _parse_composite_label(b)
    if ca is not None and cb is not None:
        joint_a, axis_a, val_a = ca
        joint_b, axis_b, val_b = cb

        # -- Joint identity (DOMINANT signal) --
        cost = 0.0
        if joint_a != joint_b:
            ga, gb = JOINT_TO_GROUP[joint_a], JOINT_TO_GROUP[joint_b]
            if ga == gb:
                cost += COST_SAME_REGION
            elif _MIRROR_GROUPS.get(ga) == gb:
                cost += COST_MIRROR
            elif frozenset({ga, gb}) in _ADJACENT_GROUPS:
                cost += COST_ADJACENT
            else:
                cost += COST_DISTANT

        # -- Axis (same joint, different axis → clearly different sensor) --
        if axis_a != axis_b:
            cost += COST_AXIS

        # -- Value bucket (graduated cost: sign + magnitude) --
        if val_a and val_b:
            cost += _value_bucket_cost(val_a, val_b)

        return cost

    # ------------------------------------------------------------------
    # Fallback: legacy individual-token comparison
    # ------------------------------------------------------------------
    if a in ALL_JOINTS and b in ALL_JOINTS:
        ga, gb = JOINT_TO_GROUP[a], JOINT_TO_GROUP[b]
        if ga == gb:
            return COST_SAME_REGION
        if _MIRROR_GROUPS.get(ga) == gb:
            return COST_MIRROR
        if frozenset({ga, gb}) in _ADJACENT_GROUPS:
            return COST_ADJACENT
        return COST_DISTANT

    if a in ALL_AXES and b in ALL_AXES:
        return COST_AXIS

    if _is_value_label(a) and _is_value_label(b):
        return _value_bucket_cost(a, b)

    if a in STRUCTURAL_NODES and b in STRUCTURAL_NODES:
        return 0.0

    # Cross-type substitution (should be rare)
    return 5.0


def _value_bucket_cost(val_a: str, val_b: str) -> float:
    """Graduated cost between two value-bucket labels.

    Handles the new bucket format (``"b0"``..``"b4"``, ``"n0"``..``"n4"``)
    as well as the legacy ``"pos"``/``"neg"`` labels.

    Returns 0.0 for identical labels.
    """
    if val_a == val_b:
        return 0.0

    # Legacy format: "pos"/"neg" – fall back to flat sign cost
    if val_a in ("pos", "neg") or val_b in ("pos", "neg"):
        return COST_SIGN_MISMATCH

    try:
        sign_a, bucket_a = val_a[0], int(val_a[1])
        sign_b, bucket_b = val_b[0], int(val_b[1])
    except (IndexError, ValueError):
        return COST_SIGN_MISMATCH  # unrecognised format

    cost = 0.0
    if sign_a != sign_b:
        cost += COST_SIGN_MISMATCH
    cost += abs(bucket_a - bucket_b) * COST_VALUE_STEP
    return cost


#: Default cost function used throughout the pipeline.
#: :func:`weighted_delta` assigns body-region-aware costs so that e.g.
#: swapping a left-hand joint for its right-hand mirror is cheaper than
#: swapping it for an ankle.  Set to ``None`` for unit costs.
DEFAULT_DELTA = weighted_delta


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
    """Encode value as a quantised bucket.

    Values are binned into intervals of width :data:`VALUE_BIN_WIDTH` so
    that magnitude differences contribute to the edit distance.  The sign
    is encoded as a prefix: ``b`` (positive / zero) or ``n`` (negative).

    Examples:  0.2 → ``"b0"``, 0.9 → ``"b2"``, 1.7 → ``"b4"``,
    -0.5 → ``"n1"``.
    """
    sign = "b" if value >= 0 else "n"
    bucket = min(int(abs(value) / VALUE_BIN_WIDTH), VALUE_NUM_BINS - 1)
    return f"{sign}{bucket}"


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


def _lark_tree_to_program_tree(
    tree: Tree,
    program: str,
    collapse_sensors: bool = True,
) -> ProgramTree:
    """Convert Lark parse tree to ProgramTree for edit distance.

    Args:
        tree: Lark parse tree
        program: Original program string
        collapse_sensors: If ``True`` (default), each ``sensor`` subtree is
            collapsed into a single composite node ``"joint.axis:bucket"``.
            This produces flat trees where the edit distance is driven by
            which body parts and movement magnitudes the programs use.

    Returns:
        ProgramTree with nodes and adjacency list
    """
    nodes: list[str] = []
    adj: list[list[int]] = []
    _build_simplified_tree(tree, nodes, adj, None, collapse_sensors=collapse_sensors)
    return ProgramTree(nodes=nodes, adj=adj, program=program)


def _collapse_sensor_node(
    tree: Tree, nodes: list, adj: list, parent_idx: Optional[int]
) -> Optional[int]:
    """Collapse a ``sensor`` subtree into a single composite node.

    Composite label format: ``"joint.axis:value_bucket"``.
    Example: ``"lhand.x:v0.3"``.
    """
    joint = axis = value = None
    for child in tree.children:
        if isinstance(child, Token):
            if child.type == "JOINT":
                joint = str(child)
            elif child.type == "AXIS":
                axis = str(child)
            elif child.type in ("VALUE", "NUMBER"):
                value = _normalize_value(float(child))
    if not joint or not axis:
        # Malformed sensor — fall back to plain label
        node_idx = len(nodes)
        nodes.append("sensor")
        adj.append([])
        if parent_idx is not None:
            adj[parent_idx].append(node_idx)
        return node_idx

    label = f"{joint}.{axis}"
    if value:
        label += f":{value}"

    node_idx = len(nodes)
    nodes.append(label)
    adj.append([])
    if parent_idx is not None:
        adj[parent_idx].append(node_idx)
    return node_idx


def _build_simplified_tree(
    tree: Union[Tree, Token],
    nodes: list,
    adj: list,
    parent_idx: Optional[int],
    in_sensor: bool = False,
    collapse_sensors: bool = True,
) -> Optional[int]:
    """Build simplified tree structure, skipping interval values.

    When *collapse_sensors* is ``True``, each ``sensor`` subtree becomes a
    single composite node (see :func:`_collapse_sensor_node`).  This makes
    the resulting tree ultra-flat so that UTED compares sensors directly.
    """
    if isinstance(tree, Token):
        # Skip frame indices
        if tree.type in ("FRAME", "INT"):
            return None
        if tree.type == "NUMBER" and not in_sensor:
            return None

        node_idx = len(nodes)
        if tree.type in ("VALUE", "NUMBER"):
            label = _normalize_value(float(tree.value))
        else:
            label = tree.value
        nodes.append(label)
        adj.append([])
        if parent_idx is not None:
            adj[parent_idx].append(node_idx)
        return node_idx

    # ---- Sensor collapse ------------------------------------------------
    if collapse_sensors and tree.data == "sensor":
        return _collapse_sensor_node(tree, nodes, adj, parent_idx)

    # ---- Non-sensor tree node -------------------------------------------
    node_idx = len(nodes)
    nodes.append(tree.data)
    adj.append([])
    if parent_idx is not None:
        adj[parent_idx].append(node_idx)

    child_in_sensor = in_sensor or tree.data == "sensor"
    for child in tree.children:
        _build_simplified_tree(
            child, nodes, adj, node_idx, child_in_sensor,
            collapse_sensors=collapse_sensors,
        )
    return node_idx


def parse_to_tree(
    program: str,
    collapse_sensors: bool = True,
) -> ProgramTree:
    """Parse a program string into a tree representation.

    Args:
        program: ExAct program string (e.g., ``"[0,50]lhand.x(0.3)"``).
        collapse_sensors: Collapse each sensor subtree into a single composite
            node ``"joint.axis:bucket"`` (default ``True``).  Collapsed trees
            give better separation because the edit distance focuses on
            body-part and movement-magnitude differences.

    Returns:
        ProgramTree for edit distance computation

    Raises:
        lark.exceptions.LarkError: If program cannot be parsed
    """
    tree = _PARSER.parse(program)
    return _lark_tree_to_program_tree(tree, program, collapse_sensors=collapse_sensors)


def program_edit_distance(
    program1: Union[str, ProgramTree],
    program2: Union[str, ProgramTree],
    delta: Optional[Callable] = "default",
    normalize: bool = False,
) -> float:
    """Compute unordered tree edit distance between two programs.
    
    Uses the constrained UTED algorithm from edist library.
    
    Args:
        program1: First program (string or ProgramTree)
        program2: Second program (string or ProgramTree)
        delta: Cost function.  ``"default"`` → :func:`weighted_delta`,
               ``None`` → unit costs, or a custom callable.
        normalize: If ``True``, divide the raw distance by
               ``max(len(tree1), len(tree2))`` so that the result is in
               roughly [0, max_unit_cost] regardless of program length.
               
    Returns:
        Edit distance (float >= 0)
    """
    from edist.uted import uted
    
    if delta == "default":
        delta = DEFAULT_DELTA
    
    # Parse if needed
    if isinstance(program1, str):
        program1 = parse_to_tree(program1)
    if isinstance(program2, str):
        program2 = parse_to_tree(program2)
    
    # Compute UTED
    dist = uted(
        program1.nodes, program1.adj,
        program2.nodes, program2.adj,
        delta=delta,
    )
    
    if normalize:
        max_size = max(len(program1), len(program2), 1)
        dist = dist / max_size
    
    return dist


def min_distance_to_model(
    query_program: Union[str, ProgramTree],
    model_programs: list[Union[str, ProgramTree]],
    delta: Optional[Callable] = "default",
    normalize: bool = False,
) -> tuple[float, int]:
    """Compute minimum edit distance from query to any program in model.
    
    Args:
        query_program: Query program
        model_programs: List of programs in the model
        delta: Cost function (``"default"`` → weighted, ``None`` → unit).
        normalize: Normalize each pairwise distance by max tree size.
        
    Returns:
        Tuple of (min_distance, index of closest program)
    """
    if isinstance(query_program, str):
        query_program = parse_to_tree(query_program)
    
    min_dist = float('inf')
    min_idx = -1
    
    for i, model_prog in enumerate(model_programs):
        dist = program_edit_distance(query_program, model_prog, delta, normalize=normalize)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_dist, min_idx


def batch_min_distances(
    query_programs: list[Union[str, ProgramTree]],
    model_programs: list[Union[str, ProgramTree]],
    delta: Optional[Callable] = "default",
    normalize: bool = False,
) -> list[float]:
    """Compute minimum distances for a batch of queries.
    
    Args:
        query_programs: List of query programs
        model_programs: List of programs in the model
        delta: Cost function (``"default"`` → weighted, ``None`` → unit).
        normalize: Normalize each pairwise distance by max tree size.
        
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
        dist, _ = min_distance_to_model(query, model_trees, delta, normalize=normalize)
        distances.append(dist)
    
    return distances


def _compute_row_distances(
    query_activity_idx: int,
    query_activity: str,
    query_programs_data: list[tuple[list[str], list[list[int]], str]],
    model_activities: list[str],
    model_programs_data: dict[str, list[tuple[list[str], list[list[int]], str]]],
    n_activities: int,
    use_weighted_delta: bool = True,
    normalize: bool = True,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute one row of the distance matrix (for parallel execution).
    
    Args:
        query_activity_idx: Index of query activity in matrix
        query_activity: Name of query activity
        query_programs_data: List of (nodes, adj, program) tuples for queries
        model_activities: List of all activity names
        model_programs_data: Dict mapping activity -> list of (nodes, adj, program)
        n_activities: Number of activities
        use_weighted_delta: Use :func:`weighted_delta` (True) or unit costs.
        normalize: Normalize each distance by ``max(|tree1|, |tree2|)``.
        
    Returns:
        Tuple of (row_idx, mean_distances, std_distances)
    """
    from edist.uted import uted
    
    delta = weighted_delta if use_weighted_delta else None
    
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
                    delta=delta,
                )
                if normalize:
                    max_size = max(len(query), len(model), 1)
                    dist = dist / max_size
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
        
        # Optional IDF weights per model program (set via compute_idf_weights)
        self._idf_weights: dict[str, np.ndarray] | None = None
        
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

    def compute_idf_weights(
        self,
        raw_programs_by_activity: dict[str, list[str]],
    ) -> None:
        """Compute IDF-based weights for model programs.

        Programs whose body-part tokens are *rare* across activities get
        higher weight so that the mean-sigmoid scorer emphasises
        activity-distinctive programs over generic ones.

        Call **after** :meth:`set_model_programs` and **before**
        :meth:`compute_auc_matrix`.

        Args:
            raw_programs_by_activity: ``{activity: [program_string, …]}``
                for every activity (the same strings passed to
                :meth:`set_model_programs`).
        """
        import re
        from collections import Counter

        pat = re.compile(r"([a-z_]+)\.([xyz])\(")

        # Token set per activity
        act_tokens: dict[str, set[str]] = {}
        for act in self.activity_names:
            toks: set[str] = set()
            for prog in raw_programs_by_activity.get(act, []):
                for m in pat.finditer(prog):
                    toks.add(f"{m.group(1)}.{m.group(2)}")
            act_tokens[act] = toks

        # Document frequency
        n = len(self.activity_names)
        df: Counter[str] = Counter()
        for toks in act_tokens.values():
            for t in toks:
                df[t] += 1
        idf = {t: np.log(n / c) for t, c in df.items()}

        # Per-program weight = Σ idf(token)
        self._idf_weights = {}
        for act in self.activity_names:
            weights = []
            for prog in raw_programs_by_activity.get(act, []):
                toks = [
                    f"{m.group(1)}.{m.group(2)}"
                    for m in pat.finditer(prog)
                ]
                w = sum(idf.get(t, 0.0) for t in toks) if toks else 1.0
                weights.append(max(w, 0.01))
            self._idf_weights[act] = np.array(weights)
    
    def compute_matrix(
        self,
        delta: Optional[Callable] = "default",
        verbose: bool = True,
        num_workers: int = 1,
        normalize: bool = False,
    ) -> np.ndarray:
        """Compute the separability matrix.
        
        Args:
            delta: Cost function (``"default"`` → weighted, ``None`` → unit).
            verbose: Whether to show progress
            num_workers: Number of parallel workers (default: 1 = sequential)
            normalize: Normalize distances by max tree size (default False).
            
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
                distances = batch_min_distances(
                    query_programs, model_progs, delta, normalize=normalize,
                )
                
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
                    use_weighted_delta=(DEFAULT_DELTA is not None),
                    normalize=False,
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
                w = self._idf_weights.get(model_activity) if self._idf_weights else None
                scores = _compute_sigmoid_scores(
                    query_programs, model_progs, method, weights=w,
                )
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
    weights: Optional[np.ndarray] = None,
) -> list[float]:
    """Compute sigmoid-based similarity scores for each query against a model.

    Uses :data:`DEFAULT_DELTA` (unit costs by default; set to
    :func:`weighted_delta` for body-region-aware scoring).

    Raw (un-normalised) weighted distances are transformed by a biased
    sigmoid ``σ(bias - k·d)``.

    Args:
        query_programs: Test programs to score
        model_programs: Model programs (from training)
        method: "mean-sigmoid" or "min-sigmoid"
        weights: Optional per-model-program weights for the mean-sigmoid
            method.  When provided the mean is weighted:
            ``Σ wᵢ · σ(bias - k·dᵢ) / Σ wᵢ``.  Ignored for min-sigmoid.

    Returns:
        List of scores, one per query program (higher = more similar to model)
    """
    from edist.uted import uted

    delta = DEFAULT_DELTA

    # Normalise weights once
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(model_programs):
            raise ValueError(
                "weights must match the number of model programs: "
                f"got {len(w)} weights for {len(model_programs)} model programs"
            )
        w = w / w.sum()
    else:
        w = None

    scores = []
    for query in query_programs:
        # Compute weighted distances to every model program.
        dists = np.array([
            uted(
                query.nodes, query.adj,
                model.nodes, model.adj,
                delta=delta,
            )
            for model in model_programs
        ])

        # Biased sigmoid: σ(bias - k·d)
        k = SIGMOID_TEMPERATURE
        b = SIGMOID_BIAS
        sigs = _sigmoid(b - k * dists)

        if method == "mean-sigmoid":
            if w is not None:
                score = float(np.dot(w, sigs))
            else:
                score = float(np.mean(sigs))
        elif method == "min-sigmoid":
            score = float(_sigmoid(np.array([b - k * dists.min()]))[0])
        else:
            raise ValueError(f"Unknown method: {method}")

        scores.append(score)

    return scores


# Need numpy for matrix operations
import numpy as np
