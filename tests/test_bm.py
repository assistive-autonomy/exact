"""Tests for BehaviourModel and HumEnv."""
import pytest
import torch

from exact.env import HumEnv
from exact.bm import BehaviourModel
from exact.programs import parse_program
from exact.programs.rewards import sensors2reward, make_base_reward, SensorReward


# Skip tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestHumEnv:
    """Tests for HumEnv wrapper."""

    @pytest.fixture
    def env(self):
        return HumEnv(device="cuda" if torch.cuda.is_available() else "cpu")

    def test_env_initialization(self, env):
        """Test environment initializes correctly."""
        assert env is not None
        assert env.model is not None
        assert env.data is not None

    def test_env_reset(self, env):
        """Test environment reset returns correct observation shape."""
        obs, info = env.reset()
        assert obs.shape == (1, 358)
        assert obs.dtype == torch.float32

    def test_env_step(self, env):
        """Test environment step with random action."""
        env.reset()
        action = torch.zeros(56).numpy()
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == (1, 358)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_env_render(self, env):
        """Test environment render returns frame."""
        env.reset()
        frame = env.render()
        assert frame is not None
        assert len(frame.shape) == 3  # H, W, C


class TestBehaviourModel:
    """Tests for BehaviourModel."""

    @pytest.fixture
    def model(self):
        return BehaviourModel()

    @pytest.fixture
    def env(self):
        return HumEnv(device="cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def head_reward(self):
        return sensors2reward("head.z(1.4)")

    @requires_cuda
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model is not None
        assert model.model is not None

    @requires_cuda
    def test_z_from_reward(self, model, env, head_reward):
        """Test z encoding from reward function."""
        z = model.z_from_reward(env, head_reward)
        assert z is not None
        assert z.shape[0] == model.batch_size

    @requires_cuda
    def test_act(self, model, env, head_reward):
        """Test action generation."""
        obs, _ = env.reset()
        z = model.z_from_reward(env, head_reward)
        action = model.act(obs, z)
        assert action.shape == (1, 56)

    @requires_cuda
    def test_generate(self, env, model, head_reward):
        """Test motion generation."""
        steps = 50
        poses, actions = model.generate(env, head_reward, steps=steps)
        assert poses.shape[0] == steps
        assert poses.shape[1] == 358
        assert actions.shape[0] == steps
        assert actions.shape[1] == 56

    @requires_cuda
    def test_generate_with_render(self, env, model, head_reward):
        """Test motion generation with rendering."""
        steps = 10
        poses, actions, frames = model.generate(env, head_reward, steps=steps, render=True)
        assert poses.shape[0] == steps
        assert len(frames) == steps + 1  # Initial frame + step frames