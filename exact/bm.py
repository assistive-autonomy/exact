"""Motion model using pre-trained MetaMotivo."""
import h5py
import torch
from huggingface_hub import hf_hub_download
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.wrappers.humenvbench import relabel

from exact.env import HumEnv
from exact.programs import SensorReward


class BehaviourModel:
    """Generates motions from reward programs using pre-trained MetaMotivo model."""

    def __init__(
        self,
        model_name: str = "facebook/metamotivo-M-1",
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.model = FBcprModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        buffer_path = hf_hub_download(
            repo_id=model_name,
            filename="data/buffer_inference_500000.hdf5",
            repo_type="model",
        )
        with h5py.File(buffer_path, "r") as f:
            data = {k: v[:] for k, v in f.items()}
            self.buffer = DictBuffer(capacity=data["qpos"].shape[0], device=device)
            self.buffer.extend(data)

    def z_from_reward(self, env: HumEnv, reward_fn: SensorReward) -> torch.Tensor:
        """Compute latent z from reward function using buffer relabeling."""
        batch = self.buffer.sample(batch_size=self.batch_size)
        rewards = relabel(
            env.env,
            qpos=batch["qpos"].cpu(),
            qvel=batch["qvel"].cpu(),
            action=batch["action"].cpu(),
            reward_fn=reward_fn,
            max_workers=8,
        )
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        return self.model.reward_wr_inference(batch["next_observation"], rewards)

    def act(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get action from observation and latent z."""
        return self.model.act(obs, z)
