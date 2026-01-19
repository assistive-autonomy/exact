"""Tests for program edit distance module."""

import pytest
import numpy as np


class TestProgramTree:
    """Tests for program to tree conversion."""
    
    def test_parse_simple_program(self):
        from exact.programs import parse_to_tree
        
        program = "[0,50]lhand.x(0.3)"
        tree = parse_to_tree(program)
        
        assert tree is not None
        assert tree.program == program
        assert len(tree.nodes) > 0
        assert len(tree.adj) == len(tree.nodes)
    
    def test_parse_multi_motion_program(self):
        from exact.programs import parse_to_tree
        
        program = "[0,50]lhand.x(0.3)*rhand.y(-0.2);[50,100]torso.z(0.5)"
        tree = parse_to_tree(program)
        
        assert tree is not None
        # Should have start node with multiple motions
        assert "start" in tree.nodes
    
    def test_intervals_excluded(self):
        """Verify interval values (INT tokens) are not in the tree."""
        from exact.programs import parse_to_tree
        
        # Two programs with same structure but different intervals
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,100]lhand.x(0.3)"
        
        tree1 = parse_to_tree(prog1)
        tree2 = parse_to_tree(prog2)
        
        # Trees should be identical since we ignore intervals
        assert tree1.nodes == tree2.nodes
        assert tree1.adj == tree2.adj
    
    def test_value_normalization(self):
        """Verify values within tolerance map to same bucket."""
        from exact.programs.edit_distance import _normalize_value, VALUE_TOLERANCE
        
        # Values within tolerance should map to same bucket
        v1 = _normalize_value(0.3)
        v2 = _normalize_value(0.4)  # diff = 0.1 < 0.3 tolerance
        
        assert v1 == v2
        
        # Values far apart should differ
        v3 = _normalize_value(0.3)
        v4 = _normalize_value(0.8)  # diff = 0.5 > 0.3 tolerance
        
        assert v3 != v4


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
    
    def test_similar_values_low_distance(self):
        """Programs with values within tolerance should have low distance."""
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,50]lhand.x(0.4)"  # Value within tolerance
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist == 0.0  # Should match due to value bucketing
    
    def test_different_joints_positive_distance(self):
        from exact.programs import program_edit_distance
        
        prog1 = "[0,50]lhand.x(0.3)"
        prog2 = "[0,50]rhand.x(0.3)"  # Different joint
        
        dist = program_edit_distance(prog1, prog2)
        
        assert dist > 0  # Should differ
    
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
            "[0,50]rhand.x(0.3)",  # Different joint
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
