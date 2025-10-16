import unittest

from exact.bm import BehaviourModel
from exact.rewards import ArmsReward

class TestBehaviourModel(unittest.TestCase):
    def test_generate(self):
        bm = BehaviourModel()
        reward_fn = ArmsReward(stand_height=1.0)
        poses, actions = bm.generate(reward_fn, steps=50)

        self.assertEqual(poses.shape[0], 50)
        self.assertEqual(actions.shape[0], 50)