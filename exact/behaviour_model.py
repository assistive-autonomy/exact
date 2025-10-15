from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
import torch
from metamotivo.fb_cpr.huggingface import FBcprModel


class BehaviourModel:
    """Behaviour model using a pre-trained FBcprModel from MetaMotivo.
    functionality:
    - create behaviour from a a program
    - evaluate the match betweeen the behaviour and a program"""
    def __init__(self,
                 model_name: str="facebook/metamotivo-M-1", 
                 device: str = "gpu" if torch.cuda.is_available() else "cpu"):
        assert "metamotivo" in model_name, "Currently only metamotivo models are supported"
        self.device = device
        self.model_name = model_name

        self.model = FBcprModel.from_pretrained(model_name)
        self.model.to(device)
        self.env, _ = make_humenv(
            wrappers=[
                FlattenObservation,
                lambda env: TransformObservation(
                    env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space # For gymnasium <1.0.0 remove the last argument: env.observation_space
                ),
        ],
        state_init="Default",)

    def generate(self, 
                 seed: int, 
                 steps: int = 1):
        observations = []
        actions = []
        obs, _ = self.env.reset(seed=seed)
        for _ in range(steps):
            action = self.model.sample(program, obs)
            observations.append(obs.cpu())
            actions.append(action.cpu())
            obs, _, terminated, truncated, _ = self.env.step(action.cpu().numpy().squeeze())
            obs = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=self.device)
            if terminated or truncated:
                break
        return torch.cat(observations), torch.cat(actions)

    def score(self, program: str, observations: torch.Tensor, actions: torch.Tensor):
        return self.model.score(program, observations, actions)