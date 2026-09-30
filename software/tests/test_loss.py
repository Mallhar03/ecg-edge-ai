import pytest
import torch
import yaml
import os
from src.training.loss import MorphologyWeightedBCELoss

def test_loss_instantiation_valid():
    """Test that the loss function instantiates correctly with valid weights."""
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/config.yaml'))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    class_weights = cfg['model']['class_weights']
    
    loss_fn = MorphologyWeightedBCELoss(class_weights)
    assert isinstance(loss_fn, torch.nn.Module)
    assert hasattr(loss_fn, 'class_weights')
    assert loss_fn.class_weights.shape == (5,)

def test_loss_finite_and_positive():
    """Test that the loss returns a finite, positive value for typical inputs."""
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_fn = MorphologyWeightedBCELoss(weights)
    
    logits = torch.zeros(8, 5) # All 0.5 after sigmoid
    targets = torch.zeros(8, 5)
    loss_val = loss_fn(logits, targets)
    
    assert torch.isfinite(loss_val)
    assert loss_val.item() > 0

def test_loss_higher_weight_increases_loss():
    """Test that higher class weights lead to a higher loss value for the same inputs."""
    torch.manual_seed(42)
    weights_low = [1.0, 1.0, 1.0, 1.0, 1.0]
    weights_high = [5.0, 5.0, 5.0, 5.0, 5.0]
    
    loss_low_fn = MorphologyWeightedBCELoss(weights_low)
    loss_high_fn = MorphologyWeightedBCELoss(weights_high)
    
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()
    
    loss_low = loss_low_fn(logits, targets)
    loss_high = loss_high_fn(logits, targets)
    
    assert loss_high.item() > loss_low.item()

def test_loss_invalid_weights_raises():
    """Test that invalid weight configurations raise ValueError."""
    # Test wrong number of weights
    with pytest.raises(ValueError, match="class_weights must have exactly 5 elements"):
        MorphologyWeightedBCELoss([1.0, 1.0, 1.0, 1.0])
        
    # Test non-positive weights
    with pytest.raises(ValueError, match="All class_weights must be > 0"):
        MorphologyWeightedBCELoss([1.0, 1.0, 1.0, 1.0, -1.0])
    
    with pytest.raises(ValueError, match="All class_weights must be > 0"):
        MorphologyWeightedBCELoss([1.0, 0.0, 1.0, 1.0, 1.0])

def test_loss_shape_mismatch_raises():
    """Test that mismatched input shapes raise ValueError."""
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_fn = MorphologyWeightedBCELoss(weights)
    
    logits = torch.randn(8, 5)
    targets = torch.randn(8, 4) # Mismatched shape
    
    with pytest.raises(ValueError, match="logits and targets must have matching shapes"):
        loss_fn(logits, targets)
