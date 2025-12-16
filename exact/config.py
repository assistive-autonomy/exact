from typing import List, Optional
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
    num_intervals: int
    min_interval_time: int
    allowed_parts: List[str]
    render: bool = False


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


class MotionModelConfig(BaseModel):
    name: str
    batch_size: int


class TrainingConfig(BaseModel):
    seed: int
    model: ModelConfig
    lora: LoraConfig
    motion_encoder: MotionEncoderConfig
    motion_model: MotionModelConfig
    train_data_path: str
    eval_data_path: Optional[str] = None
    grammar_path: Optional[str] = None
    num_train_epochs: int
    num_workers: int
    max_seq_length: int
    batch_size: int
    gradient_accumulation_steps: int = 4
    learning_rate: float
    weight_decay: float = 0.01
    max_grad_norm: float
    warmup_steps: int
    save_every: int = 5


class WandbConfig(BaseModel):
    project: str
    entity: str
    name: str
    mode: str


class ExperimentConfig(BaseModel):
    training: TrainingConfig
    wandb: WandbConfig
