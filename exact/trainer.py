"""Trainer module for motion-conditioned parser."""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ParserTrainer:
    """Trainer for MotionConditionedParser."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
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
            poses = batch["poses"].to(self.device)

            # Labels are input_ids shifted (model handles internally)
            labels = input_ids.clone()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                motion=poses,
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
        logger=None,
    ) -> dict:
        """Evaluate the model.
        
        Args:
            eval_loader: DataLoader for evaluation data
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

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            poses = batch["poses"].to(self.device)
            labels = input_ids.clone()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                motion=poses,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        if logger is not None:
            logger.log({
                "eval/loss": avg_loss,
                "eval/step": self.global_step,
            })

        return {
            "loss": avg_loss,
        }

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