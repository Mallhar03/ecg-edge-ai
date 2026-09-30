#!/usr/bin/env python3
"""
Train ECG-FPGA Accelerator Model.

Usage:
    python software/scripts/train.py --config software/config/config.yaml
    python software/scripts/train.py --config software/config/config.yaml --resume software/outputs/checkpoints/best_model.pth

This script:
1. Loads data using inter-patient splitting (AAMI standard, no data leakage)
2. Builds the enhanced MultiScale1DCNN model
3. Configures focal loss + cosine LR scheduling
4. Trains with early stopping and checkpointing
5. Optimizes per-class thresholds on validation set
"""

import argparse
import yaml
import sys
import os
import logging
import torch

# Ensure software directory is in path for imports
sys.path.append(os.path.join(os.getcwd(), 'software'))

from src.data.dataset import make_dataloaders
from src.models.multiscale_cnn import MultiScale1DCNN
from src.training.loss import build_loss_fn
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train ECG-FPGA Accelerator Model")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--resume", help="Path to checkpoint .pth file to resume from")
    args = parser.parse_args()

    # Logging setup for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("ecg_edge.scripts.train")

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # ── Data: Inter-patient split (BUG 1 FIX) ──
        logger.info("Loading data with inter-patient splitting...")
        train_loader, val_loader, _ = make_dataloaders(config)

        # ── Model: Enhanced architecture ──
        model = MultiScale1DCNN(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {total_params:,} parameters")

        # ── Loss: Focal or BCE from config ──
        loss_fn = build_loss_fn(config, device=device)

        # ── Trainer ──
        trainer = Trainer(config)

        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            optimizer = trainer._get_optimizer(model)
            trainer.load_checkpoint(args.resume, model, optimizer)

        trainer.fit(model, train_loader, val_loader, loss_fn)
        logger.info("Training complete. Best model saved to outputs/checkpoints/best_model.pth")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
