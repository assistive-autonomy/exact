"""Tests for grammar validation and program repair utilities."""

import pytest

from exact.parser.utils import (
    validate_program,
    repair_program,
    post_process_program,
    extract_valid_prefix,
    get_grammar_parser,
)


class TestGrammarParser:
    """Tests for the Lark grammar parser."""

    def test_parser_creation(self):
        """Test that the grammar parser can be created."""
        parser = get_grammar_parser()
        assert parser is not None

    def test_valid_single_motion(self):
        """Test parsing a single motion segment."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,20]head.y(1.65)")
        assert tree is not None

    def test_valid_multi_sensor(self):
        """Test parsing motion with multiple sensors."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,50]head.y(1.65)*rhand.z(0.45)")
        assert tree is not None

    def test_valid_multi_motion(self):
        """Test parsing multiple motion segments."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,50]head.y(1.65);[50,100]pelvis.y(0.85)")
        assert tree is not None

    def test_valid_negative_value(self):
        """Test parsing sensor with negative value."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,20]head.y(-1.5)")
        assert tree is not None

    def test_valid_decimal_value(self):
        """Test parsing sensor with decimal value."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,20]head.y(0.123)")
        assert tree is not None

    def test_valid_integer_value(self):
        """Test parsing sensor with integer value."""
        parser = get_grammar_parser()
        tree = parser.parse("[0,20]head.y(1)")
        assert tree is not None


class TestValidateProgram:
    """Tests for validate_program function."""

    def test_valid_simple(self):
        """Test validation of a simple valid program."""
        assert validate_program("[0,20]head.y(1.65)") is True

    def test_valid_complex(self):
        """Test validation of a complex valid program."""
        assert validate_program("[0,30]head.y(1.65)*rhand.z(0.45);[30,60]pelvis.y(0.80)") is True

    def test_invalid_missing_bracket(self):
        """Test validation catches missing bracket."""
        assert validate_program("0,20]head.y(1.65)") is False

    def test_invalid_wrong_joint(self):
        """Test validation catches invalid joint name."""
        assert validate_program("[0,20]invalid_joint.y(1.65)") is False

    def test_invalid_wrong_axis(self):
        """Test validation catches invalid axis."""
        assert validate_program("[0,20]head.w(1.65)") is False

    def test_invalid_empty(self):
        """Test validation catches empty string."""
        assert validate_program("") is False

    def test_valid_all_joints(self):
        """Test that all supported joints are valid."""
        joints = [
            "pelvis", "torso", "spine", "chest", "neck", "head",
            "lhip", "lknee", "lankle", "ltoe",
            "rhip", "rknee", "rankle", "rtoe",
            "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
            "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
        ]
        for joint in joints:
            program = f"[0,10]{joint}.x(1.0)"
            assert validate_program(program) is True, f"Joint '{joint}' should be valid"

    def test_valid_with_whitespace(self):
        """Programs with whitespace should be rejected by validate but accepted after repair."""
        # validate_program rejects whitespace (fast char-whitelist check)
        assert validate_program("[0, 20] head.y(1.65)") is False
        # But repair_program + validate should work
        from exact.parser.utils import repair_program
        repaired = repair_program("[0, 20] head.y(1.65)")
        assert validate_program(repaired) is True


class TestRepairProgram:
    """Tests for repair_program function."""

    def test_repair_strips_whitespace(self):
        """Test that repair strips leading/trailing whitespace."""
        assert repair_program("  [0,20]head.y(1.65)  ") == "[0,20]head.y(1.65)"

    def test_repair_removes_internal_spaces(self):
        """Test that repair removes internal spaces."""
        assert repair_program("[0, 20] head.y(1.65)") == "[0,20]head.y(1.65)"

    def test_repair_preserves_valid(self):
        """Test that repair preserves already valid programs."""
        original = "[0,20]head.y(1.65)"
        assert repair_program(original) == original

    def test_repair_complex_spacing(self):
        """Test repair with complex spacing issues."""
        result = repair_program("[ 0 , 20 ] head . y ( 1.65 )")
        assert result == "[0,20]head.y(1.65)"


class TestPostProcessProgram:
    """Tests for post_process_program function."""

    def test_valid_unchanged(self):
        """Test that valid programs are returned unchanged."""
        program = "[0,20]head.y(1.65)"
        result, is_valid = post_process_program(program)
        assert result == program
        assert is_valid is True

    def test_repair_fixes_spacing(self):
        """Test that spacing programs are valid (grammar ignores whitespace)."""
        # With the updated grammar that ignores whitespace, spaced programs are valid
        result, is_valid = post_process_program("[0, 20] head.y(1.65)")
        assert is_valid is True  # Valid because grammar ignores whitespace

    def test_invalid_not_repaired(self):
        """Test handling of invalid programs that can't be repaired."""
        result, is_valid = post_process_program("completely invalid")
        assert is_valid is False


class TestExtractValidPrefix:
    """Tests for extract_valid_prefix function."""

    def test_full_valid_program(self):
        """Test that fully valid programs return unchanged."""
        program = "[0,20]head.y(1.65);[20,40]pelvis.y(0.9)"
        result = extract_valid_prefix(program)
        assert result == program

    def test_partial_valid(self):
        """Test extraction of valid prefix from partially valid program."""
        # First segment is valid, second is malformed
        program = "[0,20]head.y(1.65);[20,40]invalid"
        result = extract_valid_prefix(program)
        assert result == "[0,20]head.y(1.65)"

    def test_all_invalid(self):
        """Test that completely invalid programs return empty string."""
        result = extract_valid_prefix("completely invalid")
        assert result == ""


class TestNumberTerminal:
    """Tests for FRAME/VALUE terminals (frame indices are integers, sensor values are floats)."""

    def test_zero_in_brackets(self):
        """Test that [0,X] parses correctly."""
        assert validate_program("[0,20]head.y(1.0)") is True

    def test_integer_time_indices(self):
        """Test that integer time indices work."""
        assert validate_program("[0,100]head.y(1.0)") is True

    def test_integer_value_in_sensor(self):
        """Test that integer sensor value parses correctly."""
        assert validate_program("[10,20]head.y(0)") is True

    def test_decimal_value(self):
        """Test that decimal value parses correctly."""
        assert validate_program("[10,20]head.y(0.5)") is True

    def test_negative_value(self):
        """Test that negative value parses correctly."""
        assert validate_program("[10,20]head.y(-0.5)") is True

    def test_mixed_numbers(self):
        """Test program with various number formats."""
        assert validate_program("[0,50]head.y(0.0)*pelvis.z(-0.5);[50,100]chest.x(0)") is True

    def test_simple_integer_value(self):
        """Test that simple integer like head.y(1) parses correctly."""
        assert validate_program("[0,30]head.y(1)") is True

    def test_float_frame_rejected(self):
        """Frame indices must be non-negative integers, not floats."""
        assert validate_program("[0.5,30]head.y(1)") is False

    def test_negative_frame_rejected(self):
        """Frame indices must be non-negative integers."""
        assert validate_program("[-1,30]head.y(1)") is False
