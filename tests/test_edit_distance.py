"""Tests for program edit distance module."""

import pytest
import numpy as np


class TestWeightedDelta:
    """Tests for the body-region-aware weighted cost function.

    The delta operates primarily on **composite sensor labels**
    (e.g. ``"lhand.x:pos"``) produced by the collapsed tree builder.
    Legacy individual-token labels are also supported for backward compat.
    """

    # ── Composite sensor labels ──────────────────────────────────────────

    def test_identical_composite_zero_cost(self):
        from exact.programs.edit_distance import weighted_delta
        assert weighted_delta("lhand.x:pos", "lhand.x:pos") == 0.0

    def test_same_region_composite(self):
        from exact.programs.edit_distance import weighted_delta, COST_SAME_REGION
        # lhand <-> lwrist = same region (left_arm), same axis & sign
        cost = weighted_delta("lhand.x:pos", "lwrist.x:pos")
        assert cost == COST_SAME_REGION

    def test_mirror_composite(self):
        from exact.programs.edit_distance import weighted_delta, COST_MIRROR
        cost = weighted_delta("lhand.x:pos", "rhand.x:pos")
        assert cost == COST_MIRROR

    def test_adjacent_region_composite(self):
        from exact.programs.edit_distance import weighted_delta, COST_ADJACENT
        # lhand <-> lhip = adjacent (left_arm <-> left_leg)
        cost = weighted_delta("lhand.x:pos", "lhip.x:pos")
        assert cost == COST_ADJACENT

    def test_distant_region_composite(self):
        from exact.programs.edit_distance import weighted_delta, COST_DISTANT
        cost = weighted_delta("lhand.x:pos", "rankle.x:pos")
        assert cost == COST_DISTANT

    def test_axis_mismatch_adds_cost(self):
        from exact.programs.edit_distance import weighted_delta, COST_AXIS
        same_axis = weighted_delta("lhand.x:pos", "rhand.x:pos")
        diff_axis = weighted_delta("lhand.x:pos", "rhand.y:pos")
        assert diff_axis - same_axis == pytest.approx(COST_AXIS)

    def test_sign_mismatch_adds_cost(self):
        from exact.programs.edit_distance import weighted_delta, COST_SIGN_MISMATCH
        same_sign = weighted_delta("lhand.x:pos", "lhand.x:pos")
        diff_sign = weighted_delta("lhand.x:pos", "lhand.x:neg")
        assert diff_sign - same_sign == pytest.approx(COST_SIGN_MISMATCH)

    def test_same_sign_no_value_cost(self):
        from exact.programs.edit_distance import weighted_delta
        # Two positive values (regardless of magnitude) should have zero cost
        assert weighted_delta("lhand.x:pos", "lhand.x:pos") == 0.0

    def test_composite_indel_cost(self):
        from exact.programs.edit_distance import weighted_delta, COST_SENSOR_INDEL
        assert weighted_delta("lhand.x:pos", None) == COST_SENSOR_INDEL
        assert weighted_delta(None, "rankle.z:neg") == COST_SENSOR_INDEL

    def test_structural_indel_cheap(self):
        from exact.programs.edit_distance import weighted_delta, COST_STRUCTURAL_INDEL
        assert weighted_delta("start", None) == COST_STRUCTURAL_INDEL
        assert weighted_delta(None, "motion") == COST_STRUCTURAL_INDEL

    def test_structural_substitution_free(self):
        from exact.programs.edit_distance import weighted_delta
        assert weighted_delta("start", "motion") == 0.0

    # ── Legacy individual-token labels ───────────────────────────────────

    def test_legacy_joint_same_region(self):
        from exact.programs.edit_distance import weighted_delta, COST_SAME_REGION
        assert weighted_delta("lhand", "lwrist") == COST_SAME_REGION

    def test_legacy_joint_distant(self):
        from exact.programs.edit_distance import weighted_delta, COST_DISTANT
        assert weighted_delta("lhand", "rankle") == COST_DISTANT

    def test_legacy_cross_type_expensive(self):
        from exact.programs.edit_distance import weighted_delta
        assert weighted_delta("lhand", "x") == 5.0


class TestProgramTree:
    """Tests for program to tree conversion (collapsed sensors)."""
    
    def test_parse_simple_program(self):
        from exact.programs import parse_to_tree
        
        program = "[0,50]lhand.x(0.3)"
        tree = parse_to_tree(program)
        
        assert tree is not None
        assert tree.program == program
        assert len(tree.nodes) > 0
        assert len(tree.adj) == len(tree.nodes)
    
    def test_collapsed_sensor_label(self):
        """Collapsed tree should contain composite ``joint.axis:sign`` labels."""
        from exact.programs import parse_to_tree
        
        tree = parse_to_tree("[0,50]lhand.x(0.3)", collapse_sensors=True)
        # Should have: start, motion, "lhand.x:pos"
        composite = [n for n in tree.nodes if "." in str(n)]
        assert len(composite) == 1
        assert composite[0] == "lhand.x:pos"
    
    def test_multiple_sensors_collapsed(self):
        from exact.programs import parse_to_tree
        
        tree = parse_to_tree("[0,50]lhand.x(0.3)*rhand.y(0.5)", collapse_sensors=True)
        composites = [n for n in tree.nodes if "." in str(n)]
        assert len(composites) == 2
    
    def test_collapsed_tree_smaller_than_expanded(self):
        from exact.programs import parse_to_tree
        
        prog = "[0,50]lhand.x(0.3)*rhand.y(0.5)"
        collapsed = parse_to_tree(prog, collapse_sensors=True)
        expanded = parse_to_tree(prog, collapse_sensors=False)
        assert len(collapsed.nodes) < len(expanded.nodes)
    
    def test_parse_multi_motion_program(self):
        from exact.programs import parse_to_tree
        
        program = "[0,50]lhand.x(0.3)*rhand.y(-0.2);[50,100]torso.z(0.5)"
        tree = parse_to_tree(program)
        
        assert tree is not None
        assert "start" in tree.nodes
    
    def test_intervals_excluded(self):
        """Verify interval values are not in the tree."""
        from exact.programs import parse_to_tree
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,100]lhand.x(0.3)"
        
        tree1 = parse_to_tree(prog1)
        tree2 = parse_to_tree(prog2)
        
        assert tree1.nodes == tree2.nodes
        assert tree1.adj == tree2.adj
    
    def test_value_sign_encoding(self):
        """Verify values are encoded by sign only."""
        from exact.programs.edit_distance import _normalize_value
        
        # All positive values map to the same label
        assert _normalize_value(0.3) == _normalize_value(0.4)
        assert _normalize_value(0.3) == _normalize_value(5.0)
        assert _normalize_value(0.3) == "pos"
        
        # Negative values map to a different label
        assert _normalize_value(-0.3) == "neg"
        assert _normalize_value(-0.3) == _normalize_value(-1.5)
        
        # Positive != negative
        assert _normalize_value(0.3) != _normalize_value(-0.3)


class TestProgramEditDistance:
    """Tests for edit distance computation."""
    
    def test_identical_programs_zero_distance(self):
        from exact.programs import program_edit_distance
        
        program = "[0,50]lhand.x(0.3)"
        
        dist = program_edit_distance(program, program)
        
        assert dist == 0.0
    
    def test_same_structure_different_intervals_zero_distance(self):
        """Programs with same structure but different intervals should have zero distance."""
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,100]lhand.x(0.3)"  # Different interval
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist == 0.0
    
    def test_same_sign_values_zero_distance(self):
        """Programs with same-sign values should have zero distance."""
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,50]lhand.x(5.0)"  # Different magnitude, same sign
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist == 0.0  # Same sign → match
    
    def test_different_joints_positive_distance(self):
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,50]rhand.x(0.3)"  # Different joint (mirror)
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist > 0  # Should differ
    
    def test_distant_joints_higher_than_mirror(self):
        """Cross-region substitution should cost more than mirror (weighted delta)."""
        from exact.programs import program_edit_distance
        from exact.programs.edit_distance import weighted_delta
        
        base = "[0,50]lhand.x(0.3)"
        mirror = "[0,50]rhand.x(0.3)"    # mirror joint
        distant = "[0,50]lankle.x(0.3)"   # distant region
        
        dist_mirror = program_edit_distance(base, mirror, delta=weighted_delta)
        dist_distant = program_edit_distance(base, distant, delta=weighted_delta)
        
        assert dist_distant > dist_mirror
    
    def test_different_structure_higher_distance(self):
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,50]lhand.x(0.3)*rhand.y(0.5)"  # Extra sensor
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist > 0  # More structure = higher distance
    
    def test_min_distance_to_model(self):
        from exact.programs import min_distance_to_model
        
        query = "[0,50]lhand.x(0.3)"
        model = [
            "[0,50]rhand.x(0.3)",  # Mirror joint
            "[0,50]lhand.x(0.3)",  # Same (should match)
            "[0,50]torso.z(0.5)",  # Very different
        ]
        
        min_dist, min_idx = min_distance_to_model(query, model)
        
        assert min_dist == 0.0
        assert min_idx == 1  # Index of matching program
    
    def test_batch_min_distances(self):
        from exact.programs import batch_min_distances
        
        queries = [
            "[0,50]lhand.x(0.3)",
            "[0,50]rhand.y(0.5)",
        ]
        model = [
            "[0,50]lhand.x(0.3)",
            "[0,50]torso.z(0.5)",
        ]
        
        distances = batch_min_distances(queries, model)
        
        assert len(distances) == 2
        assert distances[0] == 0.0  # First query matches first model

    def test_body_region_similarity_hypothesis(self):
        """Activities using the same body region should be more similar
        (lower distance) than activities using different body regions.

        This validates the core hypothesis: edit distance with
        :func:`weighted_delta` captures inter-activity relationships
        based on body-part involvement.
        """
        from exact.programs import program_edit_distance
        from exact.programs.edit_distance import weighted_delta

        arm_prog1 = "[0,50]lhand.x(0.3)*lshoulder.y(0.5)"
        arm_prog2 = "[0,50]rhand.x(0.3)*rshoulder.y(0.5)"   # mirror arms
        leg_prog  = "[0,50]lankle.x(0.3)*lhip.y(0.5)"         # legs

        # arm ↔ arm (mirror) should be much closer than arm ↔ leg (cross-region)
        d_arm_arm = program_edit_distance(arm_prog1, arm_prog2, delta=weighted_delta)
        d_arm_leg = program_edit_distance(arm_prog1, leg_prog, delta=weighted_delta)

        assert d_arm_arm < d_arm_leg, (
            f"Mirror-arm distance ({d_arm_arm:.2f}) should be less than "
            f"arm-leg distance ({d_arm_leg:.2f})"
        )


class TestProgramDistanceMatrix:
    """Tests for separability matrix computation."""
    
    def test_matrix_shape(self):
        from exact.programs import ProgramDistanceMatrix
        
        activities = ["Grab", "Put", "Carry"]
        matrix_calc = ProgramDistanceMatrix(activities)
        
        # Set model programs
        matrix_calc.set_model_programs("Grab", ["[0,50]lhand.x(0.3)"])
        matrix_calc.set_model_programs("Put", ["[0,50]rhand.y(-0.3)"])
        matrix_calc.set_model_programs("Carry", ["[0,50]torso.z(0.5)"])
        
        # Add test programs
        matrix_calc.add_test_program("Grab", "[0,50]lhand.x(0.3)")
        matrix_calc.add_test_program("Put", "[0,50]rhand.y(-0.3)")
        matrix_calc.add_test_program("Carry", "[0,50]torso.z(0.5)")
        
        # Compute
        result = matrix_calc.compute_matrix(verbose=False)
        
        assert result.shape == (3, 3)
    
    def test_diagonal_lower_than_off_diagonal(self):
        """Test that same-activity distance is lower than cross-activity."""
        from exact.programs import ProgramDistanceMatrix
        
        activities = ["A", "B"]
        matrix_calc = ProgramDistanceMatrix(activities)
        
        # Activity A: left hand movements
        matrix_calc.set_model_programs("A", [
            "[0,50]lhand.x(0.3)",
            "[0,50]lhand.y(0.3)",
        ])
        
        # Activity B: right hand movements  
        matrix_calc.set_model_programs("B", [
            "[0,50]rhand.x(0.3)",
            "[0,50]rhand.y(0.3)",
        ])
        
        # Test programs similar to their models
        matrix_calc.add_test_program("A", "[0,50]lhand.x(0.3)")
        matrix_calc.add_test_program("B", "[0,50]rhand.x(0.3)")
        
        result = matrix_calc.compute_matrix(verbose=False)
        
        # Diagonal should be zero (exact match)
        assert result[0, 0] == 0.0  # A → A
        assert result[1, 1] == 0.0  # B → B
        
        # Off-diagonal should be positive (different joints)
        assert result[0, 1] > 0  # A → B
        assert result[1, 0] > 0  # B → A
    
    def test_separability_metrics(self):
        from exact.programs import ProgramDistanceMatrix
        
        activities = ["A", "B"]
        matrix_calc = ProgramDistanceMatrix(activities)
        
        # Set up simple programs
        matrix_calc.set_model_programs("A", ["[0,50]lhand.x(0.3)"])
        matrix_calc.set_model_programs("B", ["[0,50]rhand.x(0.3)"])
        
        matrix_calc.add_test_program("A", "[0,50]lhand.x(0.3)")
        matrix_calc.add_test_program("B", "[0,50]rhand.x(0.3)")
        
        matrix_calc.compute_matrix(verbose=False)
        metrics = matrix_calc.get_separability_metrics()
        
        assert "diagonal_mean" in metrics
        assert "off_diagonal_mean" in metrics
        assert "separation" in metrics
        assert "per_activity" in metrics
        
        # With this setup, separation should be positive
        assert metrics["separation"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
