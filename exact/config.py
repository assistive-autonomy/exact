from typing import Optional
from pydantic import BaseModel


class TrainConfig(BaseModel):
    seed: int

    train_data: str
    eval_data: Optional[str] = None

    model_name: str
    max_seq_length: int

    motion_dim: int
    motion_hidden_dim: int
    motion_num_layers: int
    cross_attn_every_n: int = 4
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.1

    lora_r: int
    lora_alpha: int
    lora_dropout: float

    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    report_to: str

    wandb_project: str
    wandb_entity: str
    wandb_mode: str
