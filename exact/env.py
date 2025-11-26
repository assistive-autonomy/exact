import torch

from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation

class HumEnv:
    """Wrapper for humanoid environment with tensor observations."""

    def __init__(
        self,
        device: str = "cuda",
        state_init: str = "Default",
    ):
        self.device = device

        def _obs_wrapper(env):
            return TransformObservation(
                env,
                lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device),
                env.observation_space,
            )

        self.env, _ = make_humenv(
            wrappers=[FlattenObservation, _obs_wrapper],
            state_init=state_init,
        )

    def reset(self) -> tuple[torch.Tensor, dict]:
        return self.env.reset()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        return self.env.step(action)

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def model(self):
        return self.env.unwrapped.model

    @property
    def data(self):
        return self.env.unwrapped.data
