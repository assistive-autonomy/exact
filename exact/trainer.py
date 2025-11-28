from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from syncode import SyncodeLogitsProcessor


class ParserTrainer:
    """Trainer for MotionConditionedParser."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        logger=None,
    ) -> dict:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            logger: Optional WandB or other logger
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=True,
        )

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            local_body_pos = batch["local_body_pos"].to(self.device, dtype=self.dtype)

            # Labels are input_ids shifted (model handles internally)
            labels = input_ids.clone()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                motion=local_body_pos,
                labels=labels,
            )

            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += outputs.loss.item()
            num_batches += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            progress_bar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
            })

            if logger is not None and self.global_step % 10 == 0:
                logger.log({
                    "train/loss": outputs.loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/step": self.global_step,
                })

        avg_loss = total_loss / num_batches

        return {
            "loss": avg_loss,
            "epoch": epoch,
        }

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        tokenizer=None,
        grammar_processor: SyncodeLogitsProcessor | None = None,
        logger=None,
    ) -> dict:
        """Evaluate the model.
        
        Args:
            eval_loader: DataLoader for evaluation data
            tokenizer: Tokenizer for decoding generated programs
            grammar_processor: SyncodeLogitsProcessor for grammar-constrained generation
            logger: Optional logger
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            eval_loader,
            desc="Evaluating",
            leave=False,
        )

        num_valid_programs = 0
        total_samples = 0
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            local_body_pos = batch["local_body_pos"].to(self.device, dtype=self.dtype)
            labels = input_ids.clone()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                motion=local_body_pos,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1
            total_samples += input_ids.shape[0]
            
            # Generate with grammar constraints if processor is provided
            if grammar_processor is not None and tokenizer is not None:
                generated_ids = self.model.generate(
                    motion=local_body_pos,
                    max_new_tokens=128,
                    grammar_processor=grammar_processor,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # All grammar-constrained outputs are valid by construction
                num_valid_programs += generated_ids.shape[0]

            progress_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        validity_rate = num_valid_programs / total_samples if grammar_processor is not None else None

        metrics = {"loss": avg_loss}
        if validity_rate is not None:
            metrics["validity_rate"] = validity_rate
        
        if logger is not None:
            log_dict = {
                "eval/loss": avg_loss,
                "eval/step": self.global_step,
            }
            if validity_rate is not None:
                log_dict["eval/validity_rate"] = validity_rate
            logger.log(log_dict)

        return metrics

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)

        return checkpoint.get("epoch", 0)