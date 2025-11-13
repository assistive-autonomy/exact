import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from exact.sft import InverseBehaviorTrainer, train_inverse_behavior_model
from exact.utils import MotionProgramDataset


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    tokenizer.batch_decode.return_value = ["test program"]
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.config.hidden_size = 768
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock the generate method
    model.generate.return_value = torch.tensor([[1, 2, 3, 1]])  # [B, seq_len]
    
    # Mock the forward pass
    outputs = MagicMock()
    outputs.loss = torch.tensor(0.5)
    model.return_value = outputs
    
    return model


@pytest.fixture
def mock_behaviour_model():
    bm = MagicMock()
    bm.model = MagicMock()
    return bm


def test_inverse_behavior_trainer_init(mock_model, mock_tokenizer, mock_behaviour_model):
    """Test that the trainer initializes correctly."""
    trainer = InverseBehaviorTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        behaviour_model=mock_behaviour_model,
        args=TrainingArguments("test_output", remove_unused_columns=False),
    )
    
    assert trainer.model == mock_model
    assert trainer.tokenizer == mock_tokenizer
    assert trainer.behaviour_model == mock_behaviour_model


def test_compute_loss(mock_model, mock_tokenizer, mock_behaviour_model):
    """Test the loss computation."""
    trainer = InverseBehaviorTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        behaviour_model=mock_behaviour_model,
        args=TrainingArguments("test_output", remove_unused_columns=False),
    )
    
    # Create test batch
    batch = {
        "motion": torch.randn(2, 10, 256),  # [batch, seq_len, motion_dim]
        "reference_program": ["test program 1", "test program 2"],
    }
    
    # Test loss computation
    loss = trainer.compute_loss(mock_model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Should be part of computation graph


def test_generate(mock_model, mock_tokenizer, mock_behaviour_model):
    """Test program generation from motion."""
    trainer = InverseBehaviorTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        behaviour_model=mock_behaviour_model,
        args=TrainingArguments("test_output", remove_unused_columns=False),
    )
    
    # Add motion projection layer for testing
    trainer.motion_projection = torch.nn.Linear(256, 768).to(mock_model.device)
    
    # Test generation
    motions = torch.randn(2, 10, 256)  # [batch, seq_len, motion_dim]
    programs = trainer.generate(motions)
    
    assert isinstance(programs, list)
    assert len(programs) == 2
    assert all(isinstance(p, str) for p in programs)


@patch("exact.sft.AutoModelForCausalLM.from_pretrained")
@patch("exact.sft.AutoTokenizer.from_pretrained")
def test_train_inverse_behavior_model(
    mock_tokenizer_from_pretrained,
    mock_model_from_pretrained,
    mock_model,
    mock_tokenizer,
    mock_behaviour_model,
    tmp_path,
):
    """Test the end-to-end training function."""
    # Setup mocks
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_model_from_pretrained.return_value = mock_model
    
    # Create test data
    train_motions = [torch.randn(10, 256) for _ in range(5)]  # 5 samples
    train_programs = ["test program"] * 5
    
    # Run training
    trainer = train_inverse_behavior_model(
        train_motions=train_motions,
        train_programs=train_programs,
        val_motions=train_motions[:2],
        val_programs=train_programs[:2],
        behaviour_model=mock_behaviour_model,
        output_dir=str(tmp_path),
        model_name="test-model",
        training_config={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_program_length": 32,
        },
        wandb_enabled=False,
    )
    
    # Verify results
    assert isinstance(trainer, InverseBehaviorTrainer)
    assert trainer.model == mock_model
    assert trainer.tokenizer == mock_tokenizer
    
    # Verify model was saved
    assert (tmp_path / "pytorch_model.bin").exists()


def test_motion_program_dataset():
    """Test the MotionProgramDataset class."""
    # Create test data
    motions = [torch.randn(10, 256) for _ in range(3)]
    programs = ["program1", "program2", "program3"]
    
    # Test with programs
    dataset = MotionProgramDataset(motions, programs)
    assert len(dataset) == 3
    
    # Test item access
    item = dataset[0]
    assert "motion" in item
    assert "reference_program" in item
    assert item["reference_program"] == "program1"
    
    # Test without programs
    dataset = MotionProgramDataset(motions)
    assert len(dataset) == 3
    assert dataset[0]["reference_program"] is None
