import unittest

from exact.bm import BehaviourModel
from exact.programs.reward_simple import Head

class TestBehaviourModel(unittest.TestCase):
    def test_generate(self):
        bm = BehaviourModel()
        reward_fn = Head()
        poses, actions = bm.generate(reward_fn, steps=50)

        self.assertEqual(poses.shape[0], 50)
        self.assertEqual(actions.shape[0], 50)