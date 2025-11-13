from typing import  List
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    name: str
    seed: int
    num_samples: int
    min_units: int
    max_units: int
    min_value: float
    max_value: float
    value_step: float
    num_steps: int
    allowed_parts: List[str] = Field(
        default_factory=lambda: [
            "lhip",
            "lknee",
            "lankle",
            "ltoe",
            "rhip",
            "rknee",
            "rankle",
            "rtoe",
            "torso",
            "spine",
            "chest",
            "neck",
            "head",
            "lthorax",
            "lshoulder",
            "lelbow",
            "lwrist",
            "lhand",
            "rthorax",
            "rshoulder",
            "relbow",
            "rwrist",
            "rhand",
        ]
    )

class LoraConfig(BaseModel):    
    r: int
    lora_alpha: int 
    lora_dropout: float
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", 
                                 "v_proj",
                                 "k_proj",
                                 "o_proj",
                                 "gate_proj",
                                 "up_proj",
                                 "down_proj"])
    bias: str


class MotionEncoderConfig(BaseModel):    
    motion_dim: int
    hidden_dim: int
    num_layers: int
    num_prefix_tokens: int
    

class ModelConfig(BaseModel):
    model_name: str
    lora: LoraConfig
    motion_encoder: MotionEncoderConfig
    

class TrainingConfig(BaseModel):
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    # Loss weights
    ce_weight: float
    ot_weight: float
    # Training loop
    max_epochs: int
    gradient_clip_val: float
    # Hardware
    accelerator: str
    devices: int
    precision: str
    # Checkpointing
    checkpoint_dir: str
    save_top_k: int
    monitor: str
    # Logging
    log_every_n_steps: int
    val_check_interval: float


class InferenceConfig(BaseModel):
    num_beams: int
    temperature: float
    top_k: int
    top_p: float
    num_eval_samples: int


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
