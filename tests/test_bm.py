import pytest
from exact.bm import BehaviourModel
from exact.programs.rewards import Head


@pytest.fixture
def behaviour_model():
    """Fixture that provides a BehaviourModel instance for testing."""
    return BehaviourModel()


@pytest.fixture
def head_reward():
    """Fixture that provides a Head reward function for testing."""
    return Head()


def test_generate(behaviour_model, head_reward):
    """Test that generate returns poses and actions with expected shapes."""
    steps = 50
    poses, actions = behaviour_model.generate(head_reward, steps=steps)
    
    assert poses.shape[0] == steps
    assert actions.shape[0] == steps