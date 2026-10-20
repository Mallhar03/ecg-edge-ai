import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Logger setup
logger = logging.getLogger("ecg_edge.training.loss")


class FocalBCELoss(nn.Module):
    """
    Focal Loss for multi-label binary classification.

    Focal Loss down-weights easy-to-classify samples (mainly Normal beats)
    and focuses gradient on hard, misclassified minority beats (Ventricular,
    Supraventricular). This is more effective than pos_weight alone for
    highly imbalanced ECG datasets.

    Loss formula per sample per class:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit).
    gamma=0 reduces to standard BCE. gamma=2 is recommended.

    Args:
        class_weights (list[float]): Per-class weights of shape (num_classes,).
            Acts as alpha_t in focal loss formula.
        gamma (float): Focusing parameter. Default 2.0.
        device (torch.device): Device to place the weight tensor on.

    Raises:
        ValueError: If any weight is <= 0.
    """
    def __init__(self, class_weights, gamma=2.0, num_classes=5, device=None):
        super().__init__()
        if len(class_weights) != num_classes:
            error_msg = f"class_weights must have exactly {num_classes} elements, got {len(class_weights)}"
            raise ValueError(error_msg)
        if any(w <= 0 for w in class_weights):
            error_msg = f"All class_weights must be > 0, got {class_weights}"
            raise ValueError(error_msg)

        self.gamma = gamma
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        logger.debug(f"FocalBCELoss initialized: gamma={gamma}, weights={class_weights}")

    def forward(self, logits, targets):
        """
        Computes focal binary cross entropy loss.

        Args:
            logits: (batch_size, num_classes) - raw model output (no sigmoid applied)
            targets: (batch_size, num_classes) - ground truth labels (0.0 or 1.0)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if logits.shape != targets.shape:
            error_msg = f"logits and targets must have matching shapes, got {logits.shape} vs {targets.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Standard BCE from logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t for focal modulation
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply per-class weights
        loss = focal_weight * bce * self.class_weights.unsqueeze(0)

        return loss.mean()


class MorphologyWeightedBCELoss(nn.Module):
    """
    Binary cross-entropy loss with per-class morphology weights.

    Assigns higher training penalty to misclassification of clinically
    critical rare signal classes (Ventricular, Supraventricular) versus
    common classes (Normal). Weights loaded from config — never hardcoded.

    Args:
        class_weights (list[float]): Per-class weights of shape (num_classes,).
            Order matches config class_names.
        device (torch.device): Device to place the weight tensor on.

    Raises:
        ValueError: If any weight is <= 0.
    """
    def __init__(self, class_weights, num_classes=5, device=None):
        super().__init__()
        if len(class_weights) != num_classes:
            error_msg = f"class_weights must have exactly {num_classes} elements, got {len(class_weights)}"
            raise ValueError(error_msg)
        if any(w <= 0 for w in class_weights):
            error_msg = f"All class_weights must be > 0, got {class_weights}"
            raise ValueError(error_msg)

        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32, device=device))
        logger.debug(f"MorphologyWeightedBCELoss initialized with weights: {class_weights}")

    def forward(self, logits, targets):
        """
        Computes weighted binary cross entropy loss.

        Args:
            logits: (batch_size, num_classes) - raw model output (no sigmoid applied)
            targets: (batch_size, num_classes) - ground truth labels (0.0 or 1.0)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if logits.shape != targets.shape:
            error_msg = f"logits and targets must have matching shapes, got {logits.shape} vs {targets.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # BCEWithLogitsLoss internally applies sigmoid
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.class_weights)


class FocalCrossEntropyLoss(nn.Module):
    """
    Focal Cross-Entropy Loss for multi-class ECG classification.
    Forces competing multi-class decision boundaries across arrhythmias.
    """
    def __init__(self, class_weights, gamma=2.0, num_classes=5, device=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        logger.debug(f"FocalCrossEntropyLoss initialized: gamma={gamma}, weights={class_weights}")

    def forward(self, logits, targets):
        if targets.dim() == 2:
            target_indices = torch.argmax(targets, dim=1)
        else:
            target_indices = targets.long()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)

        weights = self.class_weights[target_indices]
        focal_weight = (1.0 - target_probs) ** self.gamma

        loss = -weights * focal_weight * target_log_probs
        return loss.mean()


def build_loss_fn(config: dict, device=None):
    """
    Factory function to build the appropriate loss function from config.
    """
    class_weights = config['model']['class_weights']
    loss_type = config['training'].get('loss_type', 'focal_ce')

    if loss_type == 'focal_ce' or loss_type == 'focal':
        gamma = config['training'].get('focal_gamma', 2.0)
        logger.info(f"Using FocalCrossEntropyLoss with gamma={gamma}, weights={class_weights}")
        return FocalCrossEntropyLoss(class_weights, gamma=gamma, device=device)
    elif loss_type == 'bce':
        logger.info(f"Using MorphologyWeightedBCELoss with weights={class_weights}")
        return MorphologyWeightedBCELoss(class_weights, device=device)
    else:
        logger.info(f"Using standard CrossEntropyLoss with weights={class_weights}")
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor)
