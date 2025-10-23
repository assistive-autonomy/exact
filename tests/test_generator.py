import pytest
from exact.programs.generator import (
    BODY_PARTS,
    GenerationConfig,
    RewardProgramGenerator,
    generate_programs,
    generate_programs_with_config
)

def test_generation_config_validation():
    """Test that GenerationConfig validates body parts correctly."""
    # Test with valid body parts
    valid_parts = ["lhip", "rknee", "head"]
    config = GenerationConfig(allowed_parts=valid_parts)
    assert config.allowed_parts == valid_parts
    
    # Test with invalid body part
    with pytest.raises(ValueError):
        GenerationConfig(allowed_parts=["invalid_part"])

def test_generate_program():
    """Test basic program generation."""
    config = GenerationConfig(
        min_units=2,
        max_units=2,  # Fixed length for testing
        min_value=0.1,
        max_value=1.0,
        value_step=0.1,
        allowed_parts=["lhip", "rknee"]
    )
    generator = RewardProgramGenerator(config)
    
    # Test multiple generations
    for _ in range(10):
        program = generator.generate_program()
        parts = program.split('*')
        assert len(parts) == 2  # Should have exactly 2 units
        
        for part in parts:
            body_part, value = part.split('-')
            assert body_part in ["lhip", "rknee"]
            value_float = float(value)
            assert 0.1 <= value_float <= 1.0
            # Check if value is a multiple of 0.1 (with floating point tolerance)
            assert abs(round(value_float * 10) / 10 - value_float) < 1e-9

def test_generate_programs():
    """Test the generate_programs convenience function."""
    num_programs = 5
    programs = generate_programs(
        num_programs=num_programs,
        min_units=1,
        max_units=3,
        min_value=0.5,
        max_value=1.0,
        value_step=0.1,
        allowed_parts=["head", "torso"]
    )
    
    assert len(programs) == num_programs
    for program in programs:
        parts = program.split('*')
        assert 1 <= len(parts) <= 3
        for part in parts:
            body_part, value = part.split('-')
            assert body_part in ["head", "torso"]
            value_float = float(value)
            assert 0.5 <= value_float <= 1.0
            assert abs(round(value_float * 10) / 10 - value_float) < 1e-9

def test_generate_programs_with_config():
    """Test generating programs with a pre-configured config."""
    config = GenerationConfig(
        min_units=3,
        max_units=3,  # Fixed length
        min_value=0.3,
        max_value=0.5,
        value_step=0.1,
        allowed_parts=["lhand", "rhand"]
    )
    
    programs = generate_programs_with_config(config, num_programs=3)
    assert len(programs) == 3
    
    for program in programs:
        parts = program.split('*')
        assert len(parts) == 3
        for part in parts:
            print(part)
            body_part, value = part.split('-')
            assert body_part in ["lhand", "rhand"]
            value_float = float(value)
            assert 0.3 <= value_float <= 0.5
            # Value should be a multiple of 0.1 between 0.3 and 0.5
            assert value_float in [0.3, 0.4, 0.5]
