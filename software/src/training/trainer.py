import os
import json
import torch
import numpy as np
import random
import logging
from tqdm import tqdm

try:
    from src.training.metrics import optimize_thresholds
except (ImportError, ModuleNotFoundError):
    from .metrics import optimize_thresholds

logger = logging.getLogger("ecg_edge.training.trainer")


class Trainer:
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Full parsed config.yaml as a dict.
        Sets up: output directories, logging, seeds from config.project.seed
        """
        self.config = config
        self.seed = config.get('project', {}).get('seed', 42)
        self._set_seeds()

        # Dynamically locate the software root directory (ecg-edge-accelerator/software)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        software_dir = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

        project_cfg = config.get('project', {})
        output_dir = project_cfg.get('output_dir', os.path.join(software_dir, "outputs"))
        if not os.path.isabs(output_dir):
            if os.path.exists(output_dir):
                output_dir = os.path.abspath(output_dir)
            else:
                candidate = os.path.join(software_dir, "outputs")
                output_dir = candidate

        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = project_cfg.get('log_dir', os.path.join(output_dir, "logs"))
        if not os.path.isabs(self.log_dir):
            self.log_dir = os.path.join(output_dir, "logs")

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
        params = list(model.parameters())
        device = params[0].device if len(params) > 0 else torch.device("cpu")

        for batch in tqdm(loader, desc="Training"):
            if len(batch) == 3:
                images, rr, labels = batch
                images = images.to(device)
                rr = rr.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images, rr)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        num_batches = len(loader) if len(loader) > 0 else 1
        return {'loss': total_loss / num_batches}

    def validate(self, model, loader, loss_fn) -> dict:
        """
        Run validation pass with no gradient computation.
        Returns: dict with keys 'loss' (float), 'f1_macro' (float),
                 'metrics' (per-class and macro-average clinical metrics),
                 'all_logits', 'all_probs', 'all_labels' (for threshold optimization)
        """
        model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        params = list(model.parameters())
        device = params[0].device if len(params) > 0 else torch.device("cpu")
        class_names = self.config.get('data', {}).get(
            'class_names',
            ['Supraventricular', 'Ventricular', 'Bundle_Branch_Block', 'Paced_Other', 'Normal']
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                if len(batch) == 3:
                    images, rr, labels = batch
                    images = images.to(device)
                    rr = rr.to(device)
                    labels = labels.to(device)
                    logits = model(images, rr)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)

                loss = loss_fn(logits, labels)

                total_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        num_batches = len(loader) if len(loader) > 0 else 1
        avg_loss = total_loss / num_batches

        if len(all_logits) == 0:
            return {
                'loss': 0.0,
                'f1_macro': 0.0,
                'metrics': {},
                'all_logits': np.empty((0, len(class_names))),
                'all_probs': np.empty((0, len(class_names))),
                'all_labels': np.empty((0, len(class_names))),
            }

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Multi-class probabilities via numerically stable softmax
        exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_indices = np.argmax(all_logits, axis=1)

        # Ensure all_labels is 2D one-hot array for robust metric calculation and threshold sweeping
        if all_labels.ndim > 1:
            true_indices = np.argmax(all_labels, axis=1)
            one_hot_labels = all_labels
        else:
            true_indices = all_labels.astype(int)
            num_cls = len(class_names)
            one_hot_labels = np.zeros((all_labels.shape[0], num_cls), dtype=np.float32)
            for idx, c in enumerate(all_labels):
                if 0 <= int(c) < num_cls:
                    one_hot_labels[idx, int(c)] = 1.0

        metrics = {}
        f1_list = []
        for c_idx, name in enumerate(class_names):
            tp = int(np.sum((true_indices == c_idx) & (pred_indices == c_idx)))
            tn = int(np.sum((true_indices != c_idx) & (pred_indices != c_idx)))
            fp = int(np.sum((true_indices != c_idx) & (pred_indices == c_idx)))
            fn = int(np.sum((true_indices == c_idx) & (pred_indices != c_idx)))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
            metrics[name] = {
                'sensitivity': float(sens),
                'specificity': float(spec),
                'precision': float(prec),
                'f1': float(f1),
                'support': int(tp + fn),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
            f1_list.append(f1)

        metrics['macro_avg'] = {
            'f1': float(np.mean(f1_list)),
            'sensitivity': float(np.mean([metrics[n]['sensitivity'] for n in class_names])),
            'specificity': float(np.mean([metrics[n]['specificity'] for n in class_names]))
        }

        return {
            'loss': avg_loss,
            'f1_macro': metrics['macro_avg']['f1'],
            'metrics': metrics,
            'all_logits': all_logits,
            'all_probs': probs,
            'all_labels': one_hot_labels,
        }

    def _get_optimizer(self, model):
        training_cfg = self.config.get('training', {})
        lr = training_cfg.get('learning_rate', 1e-3)
        opt_name = training_cfg.get('optimizer', 'adam').lower()
        wd = training_cfg.get('weight_decay', 0.0)

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
        sched_type = self.config.get('training', {}).get('lr_scheduler', 'none')

        if sched_type == 'cosine':
            min_lr = self.config.get('training', {}).get('lr_scheduler_min_lr', 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=min_lr
            )
            logger.info(f"Using CosineAnnealingLR scheduler (T_max={epochs}, eta_min={min_lr})")
            return scheduler

        elif sched_type == 'plateau':
            patience = self.config.get('training', {}).get('lr_patience', 5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=patience, min_lr=1e-6
            )
            logger.info(f"Using ReduceLROnPlateau scheduler (mode=max, factor=0.5, patience={patience})")
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

        training_cfg = self.config.get('training', {})
        epochs = training_cfg.get('epochs', 25)

        if self.scheduler is None:
            self.scheduler = self._get_scheduler(self.optimizer, epochs)

        best_f1 = -1.0
        epochs_no_improve = 0
        patience = training_cfg.get('early_stopping_patience', 10)
        checkpoint_every = training_cfg.get('checkpoint_every_n_epochs', 5)
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
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                cp_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
                self.save_checkpoint(model, self.optimizer, epoch, val_f1, cp_path)

            # Best model tracking
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
            class_names = self.config.get('data', {}).get(
                'class_names',
                ['Supraventricular', 'Ventricular', 'Bundle_Branch_Block', 'Paced_Other', 'Normal']
            )
            self.optimal_thresholds = optimize_thresholds(
                best_val_result['all_labels'],
                best_val_result['all_probs'],
                class_names=class_names,
            )
            # Save thresholds to disk
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            thresh_path = os.path.join(self.checkpoint_dir, "optimal_thresholds.json")
            with open(thresh_path, 'w') as f:
                json.dump(self.optimal_thresholds, f, indent=2)
            logger.info(f"Optimal thresholds saved to {thresh_path}: {self.optimal_thresholds}")

            # Re-save best checkpoint to embed the optimal_thresholds
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            if os.path.exists(best_path):
                cp = torch.load(best_path, map_location='cpu', weights_only=False)
                cp['optimal_thresholds'] = self.optimal_thresholds
                torch.save(cp, best_path)

    def save_checkpoint(self, model, optimizer, epoch, val_f1, path) -> None:
        """
        Save checkpoint dict containing:
            model_state_dict, optimizer_state_dict, epoch, val_f1, config
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
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

        params = list(model.parameters())
        device = params[0].device if len(params) > 0 else torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
        else:
            raise KeyError(f"No recognizable model state dict found in checkpoint: {path}")

        if optimizer and checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer = optimizer

        if 'optimal_thresholds' in checkpoint:
            self.optimal_thresholds = checkpoint['optimal_thresholds']
            logger.info(f"Loaded optimal thresholds: {self.optimal_thresholds}")

        return checkpoint.get('epoch', 0), checkpoint.get('val_f1', 0.0)
