import pytest
import numpy as np
import sys
import os

# Ensure software directory is in path for imports
sys.path.append(os.path.join(os.getcwd(), 'software'))

from src.training.metrics import compute_clinical_metrics, CLASS_NAMES

def test_metrics_perfect_predictions():
    """Test metrics with 100% correct predictions."""
    y_true = np.ones((100, 5))
    y_pred = np.ones((100, 5))
    metrics = compute_clinical_metrics(y_true, y_pred)
    
    for class_name in CLASS_NAMES:
        assert metrics[class_name]['sensitivity'] == 1.0
        assert metrics[class_name]['f1'] == 1.0

def test_metrics_all_wrong_predictions():
    """Test metrics with 100% incorrect predictions (all zeros for all ones)."""
    y_true = np.ones((100, 5))
    y_pred = np.zeros((100, 5))
    metrics = compute_clinical_metrics(y_true, y_pred)
    
    for class_name in CLASS_NAMES:
        assert metrics[class_name]['sensitivity'] == 0.0
        assert metrics[class_name]['f1'] == 0.0

def test_metrics_output_structure():
    """Test that the output dictionary has the correct keys and structure."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, (100, 5))
    y_pred = np.random.rand(100, 5)
    
    metrics = compute_clinical_metrics(y_true, y_pred)
    
    expected_keys = set(CLASS_NAMES + ['macro_avg'])
    assert set(metrics.keys()) == expected_keys
    
    for key in metrics:
        assert {'sensitivity', 'specificity', 'f1'}.issubset(set(metrics[key].keys()))

def test_metrics_invalid_shape_raises():
    """Test that mismatched or incorrect input shapes raise ValueError."""
    # Wrong number of classes
    y_true = np.random.randint(0, 2, (100, 3))
    y_pred = np.random.rand(100, 3)
    
    with pytest.raises(ValueError, match=r"Expected shape \(N, 5\)"):
        compute_clinical_metrics(y_true, y_pred)
    
    # Mismatched shapes
    y_true = np.random.randint(0, 2, (100, 5))
    y_pred = np.random.rand(101, 5)
    with pytest.raises(ValueError, match="Shapes don't match"):
        compute_clinical_metrics(y_true, y_pred)
