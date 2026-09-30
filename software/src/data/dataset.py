import logging
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from src.data.loader import load_record
from src.data.preprocessor import (
    detect_r_peaks,
    extract_windows,
    extract_labels,
    compute_normalization_stats,
)

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed ECG windows and multi-hot labels.

    Windows are normalized at access time. Pass training-set mean and std
    for consistent normalization across train/val/test splits.
    Per-window normalization is used as fallback when stats are not provided.

    IMPORTANT: Always pass the SAME mean and std (computed from training data)
    to the val and test datasets. Never recompute on val/test — that is data leakage.

    Args:
        windows: np.ndarray of shape (N, window_size). Raw (unnormalized) ECG windows.
        labels: np.ndarray of shape (N, num_classes). Multi-hot float32 labels.
        mean: Training-set global mean. If None, per-window normalization is used.
        std: Training-set global std. If None, per-window normalization is used.
        augment: If True, apply random data augmentation (training only).
        aug_config: Dict with augmentation parameters from config.
    """
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        augment: bool = False,
        aug_config: Optional[dict] = None,
    ):
        if windows.ndim != 2:
            raise ValueError(f"windows must be 2D (N, window_size), got ndim={windows.ndim}")
        if windows.shape[0] != labels.shape[0]:
            raise ValueError(f"windows and labels must have same number of samples. Got windows={windows.shape[0]}, labels={labels.shape[0]}")

        self.windows = windows
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment
        self.aug_config = aug_config or {}

        logger.info(
            f"ECGDataset initialized: {len(windows)} samples, {labels.shape[1]} classes. "
            f"Normalization: {'global stats' if mean is not None else 'per-window'}. "
            f"Augmentation: {'ON' if augment else 'OFF'}."
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].copy()

        # Data augmentation (training only)
        if self.augment:
            window = self._apply_augmentation(window)

        # Normalize
        if self.mean is not None and self.std is not None:
            normalized = (window - self.mean) / (self.std + 1e-8)
        else:
            normalized = (window - window.mean()) / (window.std() + 1e-8)

        x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        y = torch.from_numpy(self.labels[idx].astype(np.float32))
        return x, y

    def _apply_augmentation(self, window: np.ndarray) -> np.ndarray:
        """
        Apply random ECG-realistic augmentations:
        1. Random time-shift (circular shift ±N samples)
        2. Random amplitude scaling (±range %)
        3. Additive Gaussian noise
        """
        # Time shift
        shift = self.aug_config.get('time_shift_samples', 10)
        if shift > 0:
            offset = np.random.randint(-shift, shift + 1)
            window = np.roll(window, offset)

        # Amplitude scaling
        scale_range = self.aug_config.get('amplitude_scale_range', 0.1)
        if scale_range > 0:
            scale = 1.0 + np.random.uniform(-scale_range, scale_range)
            window = window * scale

        # Gaussian noise
        noise_std = self.aug_config.get('gaussian_noise_std', 0.01)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=window.shape)
            window = window + noise

        return window


def _process_records(
    record_ids: List[str],
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a list of MIT-BIH records into windows + labels.

    For each record:
        1. Load signal + annotations via loader.load_record
        2. Detect R-peaks via preprocessor.detect_r_peaks
        3. Extract fixed-size windows around R-peaks
        4. Extract multi-hot labels from annotations
        5. Filter out unmapped beats (all-zero label rows)

    Args:
        record_ids: List of MIT-BIH record ID strings.
        config: Full parsed config.yaml dict.

    Returns:
        Tuple of (all_windows, all_labels) as numpy arrays.
        all_windows: shape (N, window_size), dtype float64
        all_labels: shape (N, num_classes), dtype float32
    """
    data_cfg = config['data']
    raw_dir = os.path.join(os.getcwd(), data_cfg['raw_dir'])
    fs = data_cfg['sample_rate']
    window_size = data_cfg['window_size']
    lead_index = data_cfg['lead_index']
    physionet_db = data_cfg.get('physionet_db', 'mitdb')
    annotation_map = data_cfg['annotation_map']
    class_names = data_cfg['class_names']
    search_radius = data_cfg.get('search_radius', 12)

    all_windows = []
    all_labels = []

    for rid in record_ids:
        try:
            signal, annotation = load_record(
                rid, raw_dir, lead_index=lead_index, physionet_db=physionet_db
            )

            r_peaks = detect_r_peaks(signal, fs)
            if len(r_peaks) == 0:
                logger.warning(f"Record {rid}: No R-peaks detected, skipping.")
                continue

            windows, valid_peaks = extract_windows(signal, r_peaks, window_size)
            if len(windows) == 0:
                logger.warning(f"Record {rid}: No valid windows extracted, skipping.")
                continue

            labels = extract_labels(
                annotation, valid_peaks, annotation_map, class_names,
                search_radius=search_radius
            )

            # Filter out unmapped beats (all-zero rows from the new extract_labels)
            mapped_mask = labels.sum(axis=1) > 0
            n_unmapped = (~mapped_mask).sum()
            if n_unmapped > 0:
                logger.info(f"Record {rid}: Filtering {n_unmapped} unmapped beats.")
            windows = windows[mapped_mask]
            labels = labels[mapped_mask]

            all_windows.append(windows)
            all_labels.append(labels)

            logger.info(f"Record {rid}: {len(windows)} mapped beats extracted.")

        except Exception as e:
            logger.error(f"Failed to process record {rid}: {e}")
            raise

    if len(all_windows) == 0:
        raise RuntimeError("No windows extracted from any record. Check data path and record IDs.")

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def _make_dataloaders_base(
    config: dict,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    train_mean: float,
    train_std: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from preprocessed arrays.

    Normalization statistics must be computed from training data only and
    passed in — this function applies them uniformly to all splits.

    Args:
        config: Loaded config.yaml as dict.
        train_windows, val_windows, test_windows: Raw ECG window arrays, shape (N, window_size).
        train_labels, val_labels, test_labels: Multi-hot label arrays, shape (N, num_classes).
        train_mean: Mean computed from training windows ONLY.
        train_std: Std computed from training windows ONLY.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    aug_config = config['data'].get('augmentation', {})
    augment_train = aug_config.get('enabled', False)

    train_dataset = ECGDataset(
        train_windows, train_labels, mean=train_mean, std=train_std,
        augment=augment_train, aug_config=aug_config
    )
    val_dataset = ECGDataset(val_windows, val_labels, mean=train_mean, std=train_std)
    test_dataset = ECGDataset(test_windows, test_labels, mean=train_mean, std=train_std)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config['training']['batch_size'] * 2,
        num_workers=config['training']['num_workers'],
        pin_memory=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config['training']['batch_size'] * 2,
        num_workers=config['training']['num_workers'],
        pin_memory=False,
        drop_last=False
    )

    logger.info(f"DataLoaders ready. Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} samples.")
    return train_loader, val_loader, test_loader


def make_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders using INTER-PATIENT splitting (BUG 1 FIX).

    This function:
    1. Loads records listed in config['data']['train_records'] → train+val pool
    2. Loads records listed in config['data']['test_records'] → test set
    3. Splits train pool into train/val using val_split_fraction (patient-level)
    4. Processes each group INDEPENDENTLY — no patient beats cross split boundaries
    5. Computes normalization stats from training data ONLY

    This replaces the old random-shuffle split that caused data leakage by mixing
    beats from the same patient across train and test sets.

    Args:
        config (dict): Parsed config.yaml.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        RuntimeError: If no data could be loaded from any record.
        KeyError: If config is missing required data fields.
    """
    data_cfg = config['data']
    train_record_ids = data_cfg['train_records']
    test_record_ids = data_cfg['test_records']
    val_fraction = data_cfg.get('val_split_fraction', 0.15)

    logger.info(
        f"Inter-patient split: {len(train_record_ids)} train records, "
        f"{len(test_record_ids)} test records. Val fraction: {val_fraction}"
    )

    # ── Step 1: Split train records into train/val at the PATIENT level ──
    # Use a seeded RNG for reproducibility without affecting global state
    rng = np.random.RandomState(config['project']['seed'])
    shuffled_train_ids = list(train_record_ids)
    rng.shuffle(shuffled_train_ids)

    n_val_records = max(1, int(len(shuffled_train_ids) * val_fraction))
    val_record_ids = shuffled_train_ids[:n_val_records]
    pure_train_ids = shuffled_train_ids[n_val_records:]

    logger.info(
        f"Patient-level split: {len(pure_train_ids)} train patients, "
        f"{len(val_record_ids)} val patients, {len(test_record_ids)} test patients"
    )

    # ── Step 2: Process each group independently ──
    logger.info("Processing training records...")
    train_windows, train_labels = _process_records(pure_train_ids, config)

    logger.info("Processing validation records...")
    val_windows, val_labels = _process_records(val_record_ids, config)

    logger.info("Processing test records...")
    test_windows, test_labels = _process_records(test_record_ids, config)

    # ── Step 3: Compute normalization stats from training data ONLY ──
    train_mean, train_std = compute_normalization_stats(train_windows)

    # ── Step 4: Log class distributions for each split ──
    class_names = data_cfg['class_names']
    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = split_labels.sum(axis=0).astype(int)
        logger.info(f"{split_name} class distribution: {dict(zip(class_names, counts.tolist()))}")

    # ── Step 5: Build DataLoaders ──
    return _make_dataloaders_base(
        config,
        train_windows, train_labels,
        val_windows, val_labels,
        test_windows, test_labels,
        train_mean, train_std
    )
