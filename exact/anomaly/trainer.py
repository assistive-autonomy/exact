"""
Training utilities for STG-NF anomaly detection model.
"""

import os
import time
from typing import Optional, Callable, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_loss(nll: torch.Tensor, reduction: str = "mean") -> dict:
    """Compute loss from negative log-likelihood."""
    if reduction == "mean":
        loss = torch.mean(nll)
    elif reduction == "sum":
        loss = torch.sum(nll)
    elif reduction == "none":
        loss = nll
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return {"nll": loss, "total_loss": loss}


class Trainer:
    """
    Trainer for STG-NF model with optional wandb logging.

    Args:
        model: STG-NF model instance
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optional optimizer (default: AdamW)
        scheduler: Optional LR scheduler
        device: Computing device
        checkpoint_dir: Directory for saving checkpoints
        use_wandb: Whether to log to wandb (assumes wandb.init already called)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Default optimizer
        if optimizer is None and train_loader is not None:
            self.optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.start_epoch = 0
        self.global_step = 0

    def _log_wandb(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.use_wandb:
            wandb.log(metrics, step=step or self.global_step)

    def train(
        self,
        epochs: int = 10,
        grad_clip: float = 100.0,
        log_fn: Optional[Callable] = None,
        use_confidence: bool = False,
        log_interval: int = 100,
    ) -> dict:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            grad_clip: Gradient clipping value
            log_fn: Optional logging function (step, loss_dict) -> None
            use_confidence: Weight loss by confidence scores
            log_interval: Steps between wandb logging

        Returns:
            Training history dict
        """
        self.model.train()
        history = {"train_loss": []}

        for epoch in range(self.start_epoch, epochs):
            epoch_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for step, batch in enumerate(pbar):
                try:
                    loss = self._train_step(batch, use_confidence, grad_clip)
                    epoch_losses.append(loss)
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

                    # Log to wandb periodically
                    if self.use_wandb and step % log_interval == 0:
                        self._log_wandb(
                            {
                                "train/loss": loss,
                                "train/epoch": epoch,
                                "train/lr": self.optimizer.param_groups[0]["lr"],
                            }
                        )

                    if log_fn is not None:
                        log_fn(self.global_step, {"loss": loss})

                    self.global_step += 1

                except KeyboardInterrupt:
                    logger.warning("Training interrupted.")
                    if self.checkpoint_dir:
                        self.save_checkpoint(epoch, filename="interrupted.pth")
                    return history

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

            # Log epoch summary to wandb
            if self.use_wandb:
                self._log_wandb(
                    {
                        "epoch/train_loss": avg_loss,
                        "epoch/number": epoch + 1,
                    }
                )

            # Save checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint(epoch)

            # LR scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                logger.debug(f"LR: {lr:.2e}")
                if self.use_wandb:
                    self._log_wandb({"epoch/lr": lr})

        return history

    def _train_step(
        self, batch: tuple, use_confidence: bool, grad_clip: float
    ) -> float:
        """Single training step."""
        # Unpack batch: (pose, label, [metadata or confidence])
        pose = batch[0].to(self.device, non_blocking=True).float()
        label = batch[1].to(self.device, non_blocking=True) if len(batch) > 1 else None

        # Third element could be metadata (list) or confidence (tensor)
        confidence = None
        if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
            confidence = batch[2].to(self.device, non_blocking=True)

        # Forward pass
        z, nll = self.model(pose, label=label)

        if nll is None:
            return 0.0

        # Weight by confidence if available
        if use_confidence and confidence is not None:
            nll = nll * confidence

        loss = compute_loss(nll, reduction="mean")["total_loss"]

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(self, use_confidence: bool = False) -> torch.Tensor:
        """
        Evaluate model on test set.

        Returns:
            Anomaly scores (negative log-likelihood) for each sample
        """
        self.model.eval()
        scores = []

        pbar = tqdm(self.test_loader, desc="Testing")
        for batch in pbar:
            pose = batch[0].to(self.device, non_blocking=True).float()

            # Third element could be metadata (list) or confidence (tensor)
            confidence = None
            if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                confidence = batch[2].to(self.device, non_blocking=True)

            # Forward pass with normal label assumption
            _, nll = self.model(
                pose, label=torch.ones(pose.shape[0], device=self.device)
            )

            if use_confidence and confidence is not None:
                nll = nll * confidence

            # Higher NLL = more anomalous, so negate for "normality score"
            scores.append(-nll)

        return torch.cat(scores, dim=0).cpu()

    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"checkpoint_epoch{epoch + 1}.pth"

        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.set_actnorm_initialized()

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")
