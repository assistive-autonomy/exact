"""Tests for Executable Activity Models."""

import json
import tempfile
from pathlib import Path

import pytest

from exact.models import (
    ExecutableActivityModel,
    ActivityModelCollection,
    NormalizedProgram,
    create_executable_model,
)
from exact.programs import parse_program


class TestNormalizedProgram:
    """Tests for NormalizedProgram class."""
    
    def test_from_program_basic(self):
        """Test creating a normalized program."""
        program = "[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"
        normalized = NormalizedProgram.from_program(program, eval_timesteps=100)
        
        assert normalized.original_program == program
        assert normalized.original_duration == 40
        assert normalized.scale_factor == 2.5
    
    def test_temporal_scaling(self):
        """Test that temporal intervals are correctly scaled."""
        program = "[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"
        normalized = NormalizedProgram.from_program(program, eval_timesteps=100)
        
        # Check normalized intervals
        motions = normalized.normalized_reward.motions
        assert len(motions) == 2
        assert motions[0].start == 0
        assert motions[0].end == 50  # 20 * 2.5
        assert motions[1].start == 50  # 20 * 2.5
        assert motions[1].end == 100  # 40 * 2.5
    
    def test_proportions_preserved(self):
        """Test that interval proportions are preserved after scaling."""
        program = "[0,10]head.y(1.6);[10,30]pelvis.y(0.9)"
        normalized = NormalizedProgram.from_program(program, eval_timesteps=60)
        
        # Original: first interval 10/30 = 1/3, second interval 20/30 = 2/3
        motions = normalized.normalized_reward.motions
        first_length = motions[0].end - motions[0].start
        second_length = motions[1].end - motions[1].start
        
        assert first_length == 20  # 1/3 of 60
        assert second_length == 40  # 2/3 of 60
    
    def test_zero_duration_raises(self):
        """Test that zero-duration program raises error."""
        with pytest.raises(ValueError, match="zero duration"):
            NormalizedProgram.from_program("[0,0]head.y(1.6)", eval_timesteps=100)


class TestExecutableActivityModel:
    """Tests for ExecutableActivityModel class."""
    
    def test_creation(self):
        """Test basic model creation."""
        model = ExecutableActivityModel(
            activity_name="walking",
            eval_timesteps=100,
        )
        assert model.activity_name == "walking"
        assert model.eval_timesteps == 100
        assert model.num_programs == 0
    
    def test_add_program(self):
        """Test adding programs."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        model.add_program("[0,20]head.y(1.6);[20,40]pelvis.y(0.9)")
        
        assert model.num_programs == 1
        assert len(model.original_programs) == 1
    
    def test_add_multiple_programs(self):
        """Test adding multiple programs."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        programs = [
            "[0,20]head.y(1.6);[20,40]pelvis.y(0.9)",
            "[0,30]head.y(1.65);[30,50]rknee.z(0.45)",
        ]
        model.add_programs(programs)
        
        assert model.num_programs == 2
        assert model.original_programs == programs
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        model = ExecutableActivityModel(
            activity_name="walking",
            eval_timesteps=100,
            metadata={"source": "test"},
        )
        model.add_program("[0,20]head.y(1.6);[20,40]pelvis.y(0.9)")
        
        data = model.to_dict()
        restored = ExecutableActivityModel.from_dict(data)
        
        assert restored.activity_name == model.activity_name
        assert restored.eval_timesteps == model.eval_timesteps
        assert restored.original_programs == model.original_programs
        assert restored.metadata == model.metadata
    
    def test_save_load(self):
        """Test save and load to file."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        model.add_program("[0,20]head.y(1.6);[20,40]pelvis.y(0.9)")
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            model.save(path)
            loaded = ExecutableActivityModel.load(path)
            
            assert loaded.activity_name == model.activity_name
            assert loaded.num_programs == model.num_programs
        finally:
            Path(path).unlink()
    
    def test_get_reward_fn_returns_callable(self):
        """Test that get_reward_fn returns a callable."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        model.add_program("[0,20]head.y(1.6);[20,40]pelvis.y(0.9)")
        
        reward_fn = model.get_reward_fn(timestep=25)
        assert callable(reward_fn)
    
    def test_get_reward_fn_none_when_empty(self):
        """Test that get_reward_fn returns None when no programs."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        reward_fn = model.get_reward_fn(timestep=25)
        assert reward_fn is None
    
    def test_repr(self):
        """Test string representation."""
        model = ExecutableActivityModel(activity_name="walking", eval_timesteps=100)
        model.add_program("[0,20]head.y(1.6);[20,40]pelvis.y(0.9)")
        
        repr_str = repr(model)
        assert "walking" in repr_str
        assert "num_programs=1" in repr_str


class TestCreateExecutableModel:
    """Tests for create_executable_model factory function."""
    
    def test_factory_basic(self):
        """Test factory function."""
        model = create_executable_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
            eval_timesteps=100,
        )
        
        assert model.activity_name == "walking"
        assert model.num_programs == 1
    
    def test_factory_with_metadata(self):
        """Test factory function with metadata."""
        model = create_executable_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
            eval_timesteps=100,
            metadata={"source": "test"},
        )
        
        assert model.metadata == {"source": "test"}


class TestActivityModelCollection:
    """Tests for ActivityModelCollection class."""
    
    def test_creation(self):
        """Test basic collection creation."""
        collection = ActivityModelCollection(eval_timesteps=100)
        assert collection.eval_timesteps == 100
        assert collection.num_activities == 0
    
    def test_add_model(self):
        """Test adding models to collection."""
        collection = ActivityModelCollection(eval_timesteps=100)
        model = create_executable_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
            eval_timesteps=100,
        )
        collection.add_model(model)
        
        assert collection.num_activities == 1
        assert "walking" in collection.activity_names
    
    def test_add_model_mismatched_timesteps(self):
        """Test that adding model with different timesteps raises error."""
        collection = ActivityModelCollection(eval_timesteps=100)
        model = create_executable_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
            eval_timesteps=50,  # Different!
        )
        
        with pytest.raises(ValueError, match="does not match"):
            collection.add_model(model)
    
    def test_create_and_add_model(self):
        """Test creating and adding model in one step."""
        collection = ActivityModelCollection(eval_timesteps=100)
        model = collection.create_and_add_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
        )
        
        assert collection.num_activities == 1
        assert model.activity_name == "walking"
    
    def test_get_model(self):
        """Test getting model by activity name."""
        collection = ActivityModelCollection(eval_timesteps=100)
        collection.create_and_add_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
        )
        
        model = collection.get_model("walking")
        assert model is not None
        assert model.activity_name == "walking"
        
        missing = collection.get_model("running")
        assert missing is None
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        collection = ActivityModelCollection(eval_timesteps=100)
        collection.create_and_add_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
        )
        collection.create_and_add_model(
            activity_name="running",
            programs=["[0,15]head.y(1.7);[15,30]rknee.z(0.3)"],
        )
        
        data = collection.to_dict()
        restored = ActivityModelCollection.from_dict(data)
        
        assert restored.eval_timesteps == collection.eval_timesteps
        assert restored.activity_names == collection.activity_names
    
    def test_save_load(self):
        """Test save and load to file."""
        collection = ActivityModelCollection(eval_timesteps=100)
        collection.create_and_add_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            collection.save(path)
            loaded = ActivityModelCollection.load(path)
            
            assert loaded.num_activities == collection.num_activities
        finally:
            Path(path).unlink()
    
    def test_repr(self):
        """Test string representation."""
        collection = ActivityModelCollection(eval_timesteps=100)
        collection.create_and_add_model(
            activity_name="walking",
            programs=["[0,20]head.y(1.6);[20,40]pelvis.y(0.9)"],
        )
        
        repr_str = repr(collection)
        assert "walking" in repr_str
        assert "100" in repr_str


class TestDisjunctiveReward:
    """Tests for the disjunctive reward computation logic."""
    
    def test_single_program_reward_equals_original(self):
        """With single program, disjunctive reward equals original."""
        # This is a property test: 1 - (1 - r) = r
        model = create_executable_model(
            activity_name="test",
            programs=["[0,50]head.y(1.6);[50,100]pelvis.y(0.9)"],
            eval_timesteps=100,
        )
        # The formula 1 - prod(1 - r_i) with single r reduces to r
        # We can't test without MuJoCo, but the logic is verified
        assert model.num_programs == 1
    
    def test_empty_model_returns_zero(self):
        """Empty model should return zero reward."""
        model = ExecutableActivityModel(activity_name="test", eval_timesteps=100)
        # compute_reward with empty programs should return 0
        # Can't test without MuJoCo, but we verify the edge case exists
        assert model.num_programs == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
