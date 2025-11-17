import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from exact.bm import BehaviourModel
from exact.data import Program2PoseDataset
from exact.motion_encoder import MotionEncoder
from exact.ibm_trainer import InverseBehaviorModelTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training script for Inverse Behavior Model (IBM)."""

    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")

    # Initialize WandB
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"[INFO] WandB initialized: {wandb.run.name}")
        print(f"[INFO] WandB URL: {wandb.run.url}")

    # 1. Load tokenizer and model
    print("\n[1/6] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.training.model.tokenizer)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.model.name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    
    # Get model's hidden size for motion encoder
    model_hidden_size = model.config.hidden_size
    print(f"[INFO] Model loaded: {cfg.training.model.name}")
    print(f"[INFO] Model hidden size: {model_hidden_size}")

    # 2. Apply LoRA
    print("\n[2/6] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        lora_dropout=cfg.training.lora.dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Initialize motion encoder
    print("\n[3/6] Initializing motion encoder...")
    motion_encoder = MotionEncoder(
        motion_dim=cfg.training.motion_encoder.motion_dim,  # This should be 214 for poses only
        hidden_dim=cfg.training.motion_encoder.hidden_dim,
        output_dim=model_hidden_size,  # Must match model's hidden size
        num_layers=cfg.training.motion_encoder.num_layers,
        num_prefix_tokens=cfg.training.motion_encoder.num_prefix_tokens,
    ).to(device)
    print(f"[INFO] Motion encoder: {cfg.training.motion_encoder.motion_dim} -> {model_hidden_size}")
    print(f"[INFO] Prefix tokens: {cfg.training.motion_encoder.num_prefix_tokens}")

    # 4. Initialize behaviour model (for OT loss)
    print("\n[4/6] Loading behaviour model...")
    behaviour_model = BehaviourModel(
        model_name=cfg.training.behaviour_model.name,
        batch_size=cfg.training.behaviour_model.batch_size,
        max_episode_steps=cfg.training.behaviour_model.max_episode_steps,
        device=device,
    )
    print(f"[INFO] Behaviour model loaded: {cfg.training.behaviour_model.name}")

    # 5. Load datasets
    print("\n[5/6] Loading datasets...")
    
    train_data_path = "train.hdf5"
    eval_data_path = "eval.hdf5"
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}\n"
            "Please run: uv run scripts/generate_data.py name=train"
        )
    
    train_dataset = Program2PoseDataset(
        path=train_data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.training.mx_seq_length,
    )
    print(f"[INFO] Loaded {len(train_dataset)} training samples from {train_data_path}")
    
    eval_dataset = None
    if os.path.exists(eval_data_path):
        eval_dataset = Program2PoseDataset(
            path=eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.training.mx_seq_length,
        )
        print(f"[INFO] Loaded {len(eval_dataset)} evaluation samples from {eval_data_path}")
    else:
        print(f"[WARNING] Evaluation data not found: {eval_data_path}")
        print("[WARNING] Training without evaluation")

    # 6. Setup training arguments
    print("\n[6/6] Setting up training...")
    
    output_dir = f"outputs/ibm_seed{cfg.training.seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=cfg.training.learning_rate,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        fp16=device == "cuda",
        dataloader_num_workers=cfg.training.num_workers,
        remove_unused_columns=False,  # Important: keep all columns from dataset
        report_to="wandb" if wandb_enabled else "none",
        run_name=cfg.wandb.name if wandb_enabled else None,
    )

    # Initialize trainer
    trainer = InverseBehaviorModelTrainer(
        model=model,
        tokenizer=tokenizer,
        motion_encoder=motion_encoder,
        behaviour_model=behaviour_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ce_weight=cfg.training.ce_weight,
        ot_weight=cfg.training.ot_weight,
        max_program_length=cfg.training.mx_seq_length,
        wandb_enabled=wandb_enabled,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer.train()

    # Save final model
    print("\n[INFO] Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    motion_encoder_path = os.path.join(output_dir, "final", "motion_encoder.pt")
    torch.save(motion_encoder.state_dict(), motion_encoder_path)
    print(f"[INFO] Model saved to: {output_dir}/final")
    print(f"[INFO] Motion encoder saved to: {motion_encoder_path}")

    # Test inference on first eval example
    if eval_dataset and len(eval_dataset) > 0:
        print("\n[INFO] Testing inference on first eval example...")
        test_sample = eval_dataset[0]
        test_poses = test_sample["poses"].unsqueeze(0).to(device)  # [1, T, 214]
        
        with torch.no_grad():
            motion_embeddings = motion_encoder(test_poses)
            generated_ids = model.generate(
                inputs_embeds=motion_embeddings,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            generated_program = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        reference_program = tokenizer.decode(test_sample["input_ids"], skip_special_tokens=True)
        
        print(f"Reference program:  {reference_program}")
        print(f"Generated program:  {generated_program}")
        
        if wandb_enabled:
            wandb.log({
                "test/reference_program": reference_program,
                "test/generated_program": generated_program,
            })

    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print("=" * 80)

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
