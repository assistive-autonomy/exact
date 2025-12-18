"""Configuration dataclasses for EXACT."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for parser training (maps to HuggingFace TrainingArguments)."""
    # Random seed
    seed: int = 42
    
    # Data paths
    train_data: str = "train.h5"
    eval_data: Optional[str] = "eval.h5"
    
    # Model
    model_name: str = "/pvc/phi-2"
    max_seq_length: int = 512
    
    # Motion encoder
    motion_dim: int = 72  # 24 joints * 3 (x, y, z)
    motion_hidden_dim: int = 256
    motion_num_layers: int = 6
    num_prefix_tokens: int = 8
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Training (HuggingFace TrainingArguments)
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    bf16: bool = True
    report_to: str = "wandb"
    
    # Logging
    wandb_project: str = "exact"
    wandb_entity: str = "assistive-autonomy"
    wandb_mode: str = "disabled"

