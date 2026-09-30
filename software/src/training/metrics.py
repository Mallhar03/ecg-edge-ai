import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("ecg_edge.training.metrics")

# Default class names (AAMI EC57 standard mapping)
CLASS_NAMES = ["Supraventricular", "Ventricular", "Bundle_Branch_Block", "Paced_Other", "Normal"]

def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute per-class and macro-average clinical metrics.

    Args:
        y_true (np.ndarray): Ground truth labels, shape (N, C), values 0 or 1.
        y_pred_probs (np.ndarray): Predicted probabilities after sigmoid,
            shape (N, C), values in [0, 1].
        threshold (float): Decision threshold for binary classification.
        class_names (list[str], optional): Class names for keys in output dict.
            Defaults to CLASS_NAMES if None.

    Returns:
        dict: Keys are class names + 'macro_avg'. Each value is a dict with:
            'sensitivity' (float): TP / (TP + FN). Also called recall.
            'specificity' (float): TN / (TN + FP).
            'f1' (float): 2 * precision * recall / (precision + recall).
            'precision' (float): TP / (TP + FP).
            'support' (int): Number of positive samples for this class.
    Raises:
        ValueError: If shapes don't match or y_true is not 2D.
    """
    if y_true.shape != y_pred_probs.shape:
        error_msg = f"Shapes don't match: y_true {y_true.shape}, y_pred_probs {y_pred_probs.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(y_true.shape) != 2:
        error_msg = f"Expected shape (N, C), got {y_true.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    num_classes = y_true.shape[1]

    if class_names is None:
        class_names = CLASS_NAMES

    if len(class_names) != num_classes:
        error_msg = f"class_names length ({len(class_names)}) != num_classes ({num_classes})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    y_pred = (y_pred_probs >= threshold).astype(int)

    metrics = {}

    sensitivities = []
    specificities = []
    f1s = []

    for i, class_name in enumerate(class_names):
        tp = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        tn = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))
        fp = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))

        support = tp + fn

        # Sensitivity (Recall)
        if (tp + fn) == 0:
            logger.warning(f"Class {class_name} has zero positive samples. Setting sensitivity to 0.0")
            sensitivity = 0.0
        else:
            sensitivity = tp / (tp + fn)

        # Specificity
        if (tn + fp) == 0:
            logger.warning(f"Class {class_name} has zero negative samples. Setting specificity to 0.0")
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 Score
        recall = sensitivity
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metrics[class_name] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
            'support': int(support),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)

    metrics['macro_avg'] = {
        'sensitivity': float(np.mean(sensitivities)),
        'specificity': float(np.mean(specificities)),
        'f1': float(np.mean(f1s))
    }

    return metrics


def optimize_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_thresholds: int = 100,
) -> dict:
    """
    Find per-class optimal thresholds that maximize F1 on the validation set.

    Instead of using a flat threshold=0.5 for all classes, this sweeps
    thresholds from 0.01 to 0.99 per class and picks the one with the
    highest F1 score.

    Args:
        y_true: Ground truth labels, shape (N, C).
        y_pred_probs: Predicted probabilities, shape (N, C).
        class_names: Optional list of class names for logging.
        n_thresholds: Number of threshold steps to sweep.

    Returns:
        Dict mapping class_name -> optimal_threshold (float).
    """
    num_classes = y_true.shape[1]
    if class_names is None:
        class_names = CLASS_NAMES[:num_classes]

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    optimal = {}

    for i, class_name in enumerate(class_names):
        best_f1 = -1.0
        best_t = 0.5

        y_true_i = y_true[:, i]
        y_prob_i = y_pred_probs[:, i]

        for t in thresholds:
            y_pred_i = (y_prob_i >= t).astype(int)
            tp = np.sum((y_true_i == 1) & (y_pred_i == 1))
            fp = np.sum((y_true_i == 0) & (y_pred_i == 1))
            fn = np.sum((y_true_i == 1) & (y_pred_i == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        optimal[class_name] = best_t
        logger.info(f"Optimal threshold for {class_name}: {best_t:.3f} (F1={best_f1:.4f})")

    return optimal
