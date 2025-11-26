"""Tests for reward functions and program parsing."""
import pytest
import torch

from exact.programs.rewards import (
    BODY_NAMES,
    AXIS_INDEX,
    make_base_reward,
    SensorReward,
    MotionReward,
    Reward,
    parse_program,
    sensors2reward,
)


class TestBodyMappings:
    """Tests for body name and axis mappings."""

    def test_all_body_parts_mapped(self):
        """Verify all 23 body parts are mapped."""
        expected_parts = {
            "pelvis", "torso", "spine", "chest", "neck", "head",
            "lhip", "lknee", "lankle", "ltoe",
            "rhip", "rknee", "rankle", "rtoe",
            "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
            "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
        }
        assert set(BODY_NAMES.keys()) == expected_parts
        assert len(BODY_NAMES) == 23

    def test_axis_mapping(self):
        """Test axis index mapping."""
        assert AXIS_INDEX["x"] == 0
        assert AXIS_INDEX["y"] == 1
        assert AXIS_INDEX["z"] == 2
        assert len(AXIS_INDEX) == 3


class TestMakeBaseReward:
    """Tests for make_base_reward function."""

    def test_creates_callable(self):
        """Test that make_base_reward returns a callable."""
        reward_fn = make_base_reward("head", "z", 1.4)
        assert callable(reward_fn)

    @pytest.mark.parametrize("body", list(BODY_NAMES.keys()))
    def test_all_body_parts(self, body):
        """Test reward creation for all body parts."""
        reward_fn = make_base_reward(body, "z", 1.0)
        assert callable(reward_fn)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_all_axes(self, axis):
        """Test reward creation for all axes."""
        reward_fn = make_base_reward("head", axis, 1.0)
        assert callable(reward_fn)

    @pytest.mark.parametrize("target", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_various_targets(self, target):
        """Test reward creation with various target values."""
        reward_fn = make_base_reward("head", "z", target)
        assert callable(reward_fn)


class TestSensorReward:
    """Tests for SensorReward class."""

    def test_initialization(self):
        """Test SensorReward initialization."""
        reward_fn = make_base_reward("head", "z", 1.4)
        sensor_reward = SensorReward([reward_fn])
        assert len(sensor_reward.reward_fns) == 1

    def test_multiple_reward_fns(self):
        """Test SensorReward with multiple reward functions."""
        reward_fns = [
            make_base_reward("head", "z", 1.4),
            make_base_reward("lhand", "x", 0.5),
            make_base_reward("rhand", "y", 0.5),
        ]
        sensor_reward = SensorReward(reward_fns)
        assert len(sensor_reward.reward_fns) == 3


class TestMotionReward:
    """Tests for MotionReward dataclass."""

    def test_initialization(self):
        """Test MotionReward initialization."""
        reward_fn = make_base_reward("head", "z", 1.4)
        sensor_reward = SensorReward([reward_fn])
        motion_reward = MotionReward(start=0, end=100, sensor_reward=sensor_reward)
        
        assert motion_reward.start == 0
        assert motion_reward.end == 100
        assert motion_reward.sensor_reward == sensor_reward


class TestReward:
    """Tests for Reward dataclass."""

    def test_initialization(self):
        """Test Reward initialization."""
        reward_fn = make_base_reward("head", "z", 1.4)
        sensor_reward = SensorReward([reward_fn])
        motion_reward = MotionReward(start=0, end=100, sensor_reward=sensor_reward)
        reward = Reward(motions=[motion_reward])
        
        assert len(reward.motions) == 1

    def test_get_reward_fn(self):
        """Test get_reward_fn for different timesteps."""
        reward_fn1 = make_base_reward("head", "z", 1.4)
        reward_fn2 = make_base_reward("pelvis", "y", 0.5)
        
        motion1 = MotionReward(0, 100, SensorReward([reward_fn1]))
        motion2 = MotionReward(100, 200, SensorReward([reward_fn2]))
        
        reward = Reward(motions=[motion1, motion2])
        
        # Test timestep in first motion
        fn = reward.get_reward_fn(50)
        assert fn == motion1.sensor_reward
        
        # Test timestep in second motion
        fn = reward.get_reward_fn(150)
        assert fn == motion2.sensor_reward
        
        # Test timestep outside all motions
        fn = reward.get_reward_fn(250)
        assert fn is None

    def test_get_reward_fn_boundary(self):
        """Test get_reward_fn at motion boundaries."""
        reward_fn = make_base_reward("head", "z", 1.4)
        motion = MotionReward(0, 100, SensorReward([reward_fn]))
        reward = Reward(motions=[motion])
        
        # Start boundary (inclusive)
        assert reward.get_reward_fn(0) == motion.sensor_reward
        
        # End boundary (exclusive)
        assert reward.get_reward_fn(100) is None
        
        # Just before end
        assert reward.get_reward_fn(99) == motion.sensor_reward


class TestParseProgram:
    """Tests for parse_program function."""

    def test_single_motion_single_sensor(self):
        """Test parsing single motion with one sensor."""
        program = "[0,100]head.z(1.4)"
        reward = parse_program(program)
        
        assert isinstance(reward, Reward)
        assert len(reward.motions) == 1
        assert reward.motions[0].start == 0
        assert reward.motions[0].end == 100

    def test_single_motion_multiple_sensors(self):
        """Test parsing single motion with multiple sensors."""
        program = "[0,100]head.z(1.4)*lhand.x(0.5)*rhand.y(0.5)"
        reward = parse_program(program)
        
        assert len(reward.motions) == 1
        assert len(reward.motions[0].sensor_reward.reward_fns) == 3

    def test_multiple_motions(self):
        """Test parsing multiple motions."""
        program = "[0,100]head.z(1.4),[100,200]pelvis.y(0.5)"
        reward = parse_program(program)
        
        assert len(reward.motions) == 2
        assert reward.motions[0].start == 0
        assert reward.motions[0].end == 100
        assert reward.motions[1].start == 100
        assert reward.motions[1].end == 200

    def test_negative_values(self):
        """Test parsing programs with negative values."""
        program = "[0,100]head.z(-1.4)"
        reward = parse_program(program)
        
        assert isinstance(reward, Reward)
        assert len(reward.motions) == 1

    def test_all_body_parts(self):
        """Test parsing with all body parts."""
        for body in BODY_NAMES.keys():
            program = f"[0,100]{body}.z(1.0)"
            reward = parse_program(program)
            assert len(reward.motions) == 1

    def test_all_axes(self):
        """Test parsing with all axes."""
        for axis in ["x", "y", "z"]:
            program = f"[0,100]head.{axis}(1.0)"
            reward = parse_program(program)
            assert len(reward.motions) == 1

    def test_complex_program(self):
        """Test parsing complex program with multiple intervals and sensors."""
        program = "[0,500]head.z(1.4)*lhand.x(0.5),[500,800]pelvis.y(0.8)*rhand.z(1.0)*lknee.x(-0.5),[800,1000]head.z(1.2)"
        reward = parse_program(program)
        
        assert len(reward.motions) == 3
        assert len(reward.motions[0].sensor_reward.reward_fns) == 2
        assert len(reward.motions[1].sensor_reward.reward_fns) == 3
        assert len(reward.motions[2].sensor_reward.reward_fns) == 1


class TestSensors2Reward:
    """Tests for sensors2reward helper function."""

    def test_single_sensor(self):
        """Test converting single sensor string."""
        sensor_reward = sensors2reward("head.z(1.4)")
        assert isinstance(sensor_reward, SensorReward)
        assert len(sensor_reward.reward_fns) == 1

    def test_multiple_sensors(self):
        """Test converting multiple sensors."""
        sensor_reward = sensors2reward("head.z(1.4)*lhand.x(0.5)")
        assert isinstance(sensor_reward, SensorReward)
        assert len(sensor_reward.reward_fns) == 2
