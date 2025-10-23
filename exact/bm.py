import h5py
import torch
from humenv import make_humenv
import mediapy as media
from gymnasium.wrappers import FlattenObservation, TransformObservation
from huggingface_hub import hf_hub_download
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.wrappers.humenvbench import relabel

from exact.programs import Reward


class BehaviourModel:
    """Behaviour model using a pre-trained FBcprModel from MetaMotivo."""

    def __init__(self,
                 model_name: str = "facebook/metamotivo-M-1",
                 batch_size: int = 256,
                 max_episode_steps: int = 1000,
                 buffer_small: bool = True,
                 device: str = "cpu"):
        assert "metamotivo" in model_name, "Currently only metamotivo behaviour models are supported"

        # load model (assumes model supports numpy-based inference)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.model = FBcprModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # observation wrapper: always return numpy arrays shaped (1, -1)
        def _obs_wrapper(env):
            return TransformObservation(
                env,
                lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device),
                env.observation_space,
            )

        self.env, _ = make_humenv(
            wrappers=[FlattenObservation, _obs_wrapper],
            state_init="Default",
            max_episode_steps=self.max_episode_steps,
        )

        self.buffer_name = "buffer_inference_500000.hdf5" if buffer_small else "buffer.hdf5"
        self.buffer_path = hf_hub_download(repo_id=self.model_name,
                                          filename=f"data/{self.buffer_name}",
                                          repo_type="model")
        with h5py.File(self.buffer_path, "r") as file:
            data = {k: v[:] for k, v in file.items()}
            self.buffer = DictBuffer(capacity=data["qpos"].shape[0], device=self.device)
            self.buffer.extend(data)
            del data

    def z_from_reward(self, reward_fn: Reward, max_workers: int = 8) -> torch.Tensor:
        # sample a batch from the buffer 
        batch = self.buffer.sample(batch_size=self.batch_size)
        # relabel buffer with new rewards 
        rewards = relabel(self.env,
                          qpos=batch["qpos"],
                          qvel=batch["qvel"],
                          action=batch["action"],
                          reward_fn=reward_fn,
                          max_workers=max_workers)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        # run model inference using numpy arrays (assumes model accepts numpy)
        return self.model.reward_wr_inference(batch["next_observation"], rewards)

    def generate(self,
                 reward_fn: Reward,
                 steps: int,
                 render: bool = False,
                 render_path: str = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the poses and actions taken by following the reward function (numpy, CPU)."""
        assert steps <= self.max_episode_steps, \
            f"Requested steps {steps} exceeds max episode steps {self.max_episode_steps}"
        
        obs = self.env.reset()[0] 
        z = self.z_from_reward(reward_fn)

        actions_list = []
        poses_list = []

        if render:
            frames = [self.env.render()]
            render_path = render_path or f"{reward_fn}.mp4"

        for _ in range(steps):
            # model.act expected to work with numpy arrays
            action = self.model.act(obs, z)
            obs, _, done, _, _ = self.env.step(action.cpu().numpy().ravel())

            pose = obs[:, :214]  # shape (1, 214) in observations is the pose

            poses_list.append(pose)
            actions_list.append(action)
            if render:
                frames.append(self.env.render())

            if done:
                obs = self.env.reset()[0]

        if render:
            media.write_video(render_path, frames, fps=30)
            print(f"Saved video to {render_path}")
        return torch.stack(poses_list), torch.stack(actions_list)
        