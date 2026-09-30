import os
import json
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
from src.training.metrics import compute_clinical_metrics, optimize_thresholds

logger = logging.getLogger("ecg_edge.training.trainer")

class Trainer:
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Full parsed config.yaml as a dict.
        Sets up: output directories, logging, seeds from config.project.seed
        """
        self.config = config
        self.seed = config['project']['seed']
        self._set_seeds()

        self.checkpoint_dir = os.path.join(os.getcwd(), "software/outputs/checkpoints")
        self.log_dir = os.path.join(os.getcwd(), "software/outputs/logs")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_logging()

        self.optimizer = None
        self.scheduler = None
        self.optimal_thresholds = None

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _setup_logging(self):
        log_file = os.path.join(self.log_dir, "training.log")

        # Avoid adding handlers multiple times if __init__ is called again
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler()

            formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.setLevel(logging.DEBUG)

    def train_one_epoch(self, model, loader, optimizer, loss_fn) -> dict:
        """
        Run one full training epoch.
        Returns: dict with keys 'loss' (float, mean batch loss)
        """
        model.train()
        total_loss = 0.0
        device = next(model.parameters()).device

        for images, labels in tqdm(loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / len(loader)}

    def validate(self, model, loader, loss_fn) -> dict:
        """
        Run validation pass with no gradient computation.
        Returns: dict with keys 'loss' (float), 'f1_macro' (float),
                 'metrics' (full dict from compute_clinical_metrics),
                 'all_logits', 'all_labels' (for threshold optimization)
        """
        model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        device = next(model.parameters()).device

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating"):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = loss_fn(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Apply sigmoid to logits for metrics
        probs = 1 / (1 + np.exp(-all_logits))

        class_names = self.config['data']['class_names']
        metrics = compute_clinical_metrics(all_labels, probs, class_names=class_names)

        return {
            'loss': avg_loss,
            'f1_macro': metrics['macro_avg']['f1'],
            'metrics': metrics,
            'all_probs': probs,
            'all_labels': all_labels,
        }

    def _get_optimizer(self, model):
        lr = self.config['training']['learning_rate']
        opt_name = self.config['training']['optimizer'].lower()
        wd = self.config['training']['weight_decay']

        if opt_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _get_scheduler(self, optimizer, epochs):
        """
        Build LR scheduler from config. Supports cosine annealing and
        reduce-on-plateau.
        """
        sched_type = self.config['training'].get('lr_scheduler', 'none')

        if sched_type == 'cosine':
            min_lr = self.config['training'].get('lr_scheduler_min_lr', 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr
            )
            logger.info(f"Using CosineAnnealingLR scheduler (T_max={epochs}, eta_min={min_lr})")
            return scheduler

        elif sched_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True
            )
            logger.info("Using ReduceLROnPlateau scheduler (mode=max, factor=0.5, patience=5)")
            return scheduler

        else:
            logger.info("No LR scheduler configured.")
            return None

    def fit(self, model, train_loader, val_loader, loss_fn) -> None:
        """
        Full training loop with LR scheduling, checkpointing, early stopping,
        and per-class threshold optimization.

        Saves every checkpoint_every_n_epochs to:
            software/outputs/checkpoints/epoch_{N}.pth
        Saves best model (by val f1_macro) to:
            software/outputs/checkpoints/best_model.pth
        Stops early if val f1_macro does not improve for
            config.training.early_stopping_patience consecutive epochs.
        After training, optimizes per-class thresholds on the validation set.
        """
        if self.optimizer is None:
            self.optimizer = self._get_optimizer(model)

        epochs = self.config['training']['epochs']

        if self.scheduler is None:
            self.scheduler = self._get_scheduler(self.optimizer, epochs)

        best_f1 = -1.0
        epochs_no_improve = 0
        patience = self.config['training']['early_stopping_patience']
        best_val_result = None

        for epoch in range(1, epochs + 1):
            # Get current LR for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            train_results = self.train_one_epoch(model, train_loader, self.optimizer, loss_fn)
            val_results = self.validate(model, val_loader, loss_fn)

            val_f1 = val_results['f1_macro']
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"lr: {current_lr:.6f} | "
                f"train_loss: {train_results['loss']:.4f} | "
                f"val_loss: {val_results['loss']:.4f} | "
                f"val_f1_macro: {val_f1:.4f}"
            )

            # Step LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            # Checkpoint every N epochs
            if epoch % self.config['training']['checkpoint_every_n_epochs'] == 0:
                cp_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                self.save_checkpoint(model, self.optimizer, epoch, val_f1, cp_path)

            # Best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_result = val_results
                best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                self.save_checkpoint(model, self.optimizer, epoch, val_f1, best_path)
                epochs_no_improve = 0
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # ── Per-class threshold optimization on validation set ──
        if self.config.get('inference', {}).get('optimize_thresholds', True) and best_val_result is not None:
            logger.info("Optimizing per-class thresholds on validation set...")
            class_names = self.config['data']['class_names']
            self.optimal_thresholds = optimize_thresholds(
                best_val_result['all_labels'],
                best_val_result['all_probs'],
                class_names=class_names,
            )
            # Save thresholds to disk
            thresh_path = os.path.join(self.checkpoint_dir, "optimal_thresholds.json")
            with open(thresh_path, 'w') as f:
                json.dump(self.optimal_thresholds, f, indent=2)
            logger.info(f"Optimal thresholds saved to {thresh_path}: {self.optimal_thresholds}")

    def save_checkpoint(self, model, optimizer, epoch, val_f1, path) -> None:
        """
        Save checkpoint dict containing:
            model_state_dict, optimizer_state_dict, epoch, val_f1, config
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_f1': val_f1,
            'config': self.config
        }
        if self.optimal_thresholds:
            checkpoint['optimal_thresholds'] = self.optimal_thresholds
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, model, optimizer=None):
        """
        Load checkpoint from path into model (and optionally optimizer).
        Returns: epoch number (int), val_f1 (float)
        Raises: FileNotFoundError if path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer = optimizer

        if 'optimal_thresholds' in checkpoint:
            self.optimal_thresholds = checkpoint['optimal_thresholds']
            logger.info(f"Loaded optimal thresholds: {self.optimal_thresholds}")

        return checkpoint.get('epoch', 0), checkpoint.get('val_f1', 0.0)
