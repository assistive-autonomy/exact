from typing import List
from pydantic import BaseModel


class DataConfig(BaseModel):
    name: str
    seed: int
    num_samples: int
    min_preds: int
    max_preds: int
    min_value: float
    max_value: float
    value_step: float
    num_motion_steps: int
    allowed_parts: List[str]


class LoraConfig(BaseModel):
    r: int
    alpha: int
    dropout: float


class MotionEncoderConfig(BaseModel):
    motion_dim: int
    hidden_dim: int
    num_layers: int
    num_prefix_tokens: int


class ModelConfig(BaseModel):
    name: str
    tokenizer: str


class BehaviourModelConfig(BaseModel):
    name: str
    batch_size: int
    max_episode_steps: int


class TrainingConfig(BaseModel):
    seed: int
    model: ModelConfig
    lora: LoraConfig
    motion_encoder: MotionEncoderConfig
    behaviour_model: BehaviourModelConfig
    num_train_epochs: int
    num_workers: int
    mx_seq_length: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    warmup_steps: int
    ce_weight: float
    ot_weight: float


class WandbConfig(BaseModel):
    project: str
    entity: str
    name: str
    mode: str


class ExperimentConfig(BaseModel):
    training: TrainingConfig
    wandb: WandbConfig
