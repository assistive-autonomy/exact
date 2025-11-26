import re

import pytest

from exact.programs.generator import generate_program, generate_motion, BODY_PARTS, AXES


def test_generate_motion_format():
    for _ in range(10):
        motion = generate_motion(min_preds=1, max_preds=3)
        sensors = motion.split("*")
        assert 1 <= len(sensors) <= 3

        for sensor in sensors:
            match = re.match(r"(\w+)\.([xyz])\((-?\d+\.\d+)\)", sensor)
            assert match is not None, f"Invalid format: {sensor}"
            body, axis, _ = match.groups()
            assert body in BODY_PARTS
            assert axis in AXES


def test_generate_motion_values():
    for _ in range(10):
        motion = generate_motion(min_value=0.5, max_value=1.5, value_step=0.5)
        for sensor in motion.split("*"):
            match = re.match(r"\w+\.[xyz]\((-?\d+\.\d+)\)", sensor)
            value = float(match.group(1))
            assert 0.0 <= value <= 1.5


def test_generate_motion_allowed_parts():
    allowed = ["head", "lhand"]
    for _ in range(10):
        motion = generate_motion(allowed_parts=allowed)
        for sensor in motion.split("*"):
            body = sensor.split(".")[0]
            assert body in allowed


def test_generate_program_format():
    for _ in range(10):
        program = generate_program(num_intervals=2, max_timesteps=100, min_interval_time=10)
        motions = program.split(",")
        assert len(motions) == 2

        for motion in motions:
            match = re.match(r"\[(\d+),(\d+)\](.+)", motion)
            assert match is not None, f"Invalid motion format: {motion}"
            start, end, sensors = match.groups()
            assert int(start) < int(end)


def test_generate_program_intervals_cover_timesteps():
    for _ in range(10):
        program = generate_program(num_intervals=3, max_timesteps=1000, min_interval_time=100)
        motions = program.split(",")

        times = []
        for motion in motions:
            match = re.match(r"\[(\d+),(\d+)\]", motion)
            times.append((int(match.group(1)), int(match.group(2))))

        assert times[0][0] == 0
        assert times[-1][1] == 1000
        for i in range(len(times) - 1):
            assert times[i][1] == times[i + 1][0]
