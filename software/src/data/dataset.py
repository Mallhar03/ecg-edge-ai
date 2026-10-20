import logging
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml

from src.data.loader import load_record
from src.data.preprocessor import (
    detect_r_peaks,
    extract_windows,
    extract_labels,
    compute_normalization_stats,
    compute_rr_features,
)

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed ECG windows, optional RR timing features, and multi-hot labels.

    Windows are normalized at access time. Pass training-set mean and std
    for consistent normalization across train/val/test splits.
    Per-window normalization is used as fallback when stats are not provided.

    IMPORTANT: Always pass the SAME mean and std (computed from training data)
    to the val and test datasets. Never recompute on val/test — that is data leakage.

    Args:
        windows: np.ndarray of shape (N, window_size). Raw (unnormalized) ECG windows.
        labels: np.ndarray of shape (N, num_classes). Multi-hot float32 labels.
        rr_features: Optional np.ndarray of shape (N, 2) containing normalized pre/post RR ratios.
        mean: Training-set global mean. If None, per-window normalization is used.
        std: Training-set global std. If None, per-window normalization is used.
        augment: If True, apply random data augmentation (training only).
        aug_config: Dict with augmentation parameters from config.
    """
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        rr_features: Optional[np.ndarray] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        augment: bool = False,
        aug_config: Optional[dict] = None,
    ):
        if windows.ndim not in (2, 3):
            raise ValueError(f"windows must be 2D (N, window_size) or 3D (N, C, window_size), got ndim={windows.ndim}")
        if windows.shape[0] != labels.shape[0]:
            raise ValueError(f"windows and labels must have same number of samples. Got windows={windows.shape[0]}, labels={labels.shape[0]}")
        if rr_features is not None and rr_features.shape[0] != windows.shape[0]:
            raise ValueError(f"rr_features length {rr_features.shape[0]} does not match windows {windows.shape[0]}")

        self.windows = windows
        self.labels = labels
        self.rr_features = rr_features
        self.mean = mean
        self.std = std
        self.augment = augment
        self.aug_config = aug_config or {}

        logger.info(
            f"ECGDataset initialized: {len(windows)} samples, {labels.shape[1]} classes. "
            f"RR features: {'ON' if rr_features is not None else 'OFF'}. "
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
            if window.ndim == 1:
                normalized = (window - window.mean()) / (window.std() + 1e-8)
            else:
                w_mean = window.mean(axis=-1, keepdims=True)
                w_std = window.std(axis=-1, keepdims=True)
                normalized = (window - w_mean) / (w_std + 1e-8)

        if normalized.ndim == 1:
            x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        else:
            x = torch.from_numpy(normalized.astype(np.float32))

        y = torch.from_numpy(self.labels[idx].astype(np.float32))

        if self.rr_features is not None:
            rr = torch.from_numpy(self.rr_features[idx].astype(np.float32))
            return x, rr, y

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
            window = np.roll(window, offset, axis=-1)

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess a list of MIT-BIH records into windows, labels, and RR timing features.

    For each record:
        1. Load signal + annotations via loader.load_record
        2. Detect R-peaks via preprocessor.detect_r_peaks
        3. Extract fixed-size windows around R-peaks
        4. Compute normalized pre-RR and post-RR timing ratios
        5. Extract multi-hot labels from annotations
        6. Filter out unmapped beats (all-zero label rows)

    Args:
        record_ids: List of MIT-BIH record ID strings.
        config: Full parsed config.yaml dict.

    Returns:
        Tuple of (all_windows, all_labels, all_rr) as numpy arrays.
        all_windows: shape (N, window_size), dtype float64
        all_labels: shape (N, num_classes), dtype float32
        all_rr: shape (N, 2), dtype float32
    """
    data_cfg = config['data']
    raw_dir = os.path.join(os.getcwd(), data_cfg['raw_dir'])
    fs = data_cfg['sample_rate']
    window_size = data_cfg['window_size']
    in_channels = data_cfg.get('in_channels', 1)
    if in_channels == 2:
        lead_index = [0, 1]
    else:
        lead_index = data_cfg.get('lead_index', 0)

    physionet_db = data_cfg.get('physionet_db', 'mitdb')
    annotation_map = data_cfg['annotation_map']
    class_names = data_cfg['class_names']
    search_radius = data_cfg.get('search_radius', 12)

    all_windows = []
    all_labels = []
    all_rr = []

    for rid in record_ids:
        try:
            signal, annotation = load_record(
                rid, raw_dir, lead_index=lead_index, physionet_db=physionet_db
            )

            det_signal = signal[:, 0] if signal.ndim == 2 else signal
            r_peaks = detect_r_peaks(det_signal, fs)
            if len(r_peaks) == 0:
                logger.warning(f"Record {rid}: No R-peaks detected, skipping.")
                continue

            windows, valid_peaks = extract_windows(signal, r_peaks, window_size)
            if len(windows) == 0:
                logger.warning(f"Record {rid}: No valid windows extracted, skipping.")
                continue

            rr_features = compute_rr_features(r_peaks, valid_peaks, fs=fs)

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
            rr_features = rr_features[mapped_mask]

            all_windows.append(windows)
            all_labels.append(labels)
            all_rr.append(rr_features)

            logger.info(f"Record {rid}: {len(windows)} mapped beats extracted.")

        except Exception as e:
            logger.error(f"Failed to process record {rid}: {e}")
            raise

    if len(all_windows) == 0:
        raise RuntimeError("No windows extracted from any record. Check data path and record IDs.")

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_rr, axis=0),
    )


def _make_dataloaders_base(
    config: dict,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    train_mean: float,
    train_std: float,
    train_rr: Optional[np.ndarray] = None,
    val_rr: Optional[np.ndarray] = None,
    test_rr: Optional[np.ndarray] = None,
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
        train_rr, val_rr, test_rr: Optional normalized RR timing features, shape (N, 2).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    aug_config = config['data'].get('augmentation', {})
    augment_train = aug_config.get('enabled', False)

    train_dataset = ECGDataset(
        train_windows, train_labels, rr_features=train_rr,
        mean=train_mean, std=train_std,
        augment=augment_train, aug_config=aug_config
    )
    val_dataset = ECGDataset(val_windows, val_labels, rr_features=val_rr, mean=train_mean, std=train_std)
    test_dataset = ECGDataset(test_windows, test_labels, rr_features=test_rr, mean=train_mean, std=train_std)

    balanced_sampling = config['training'].get('balanced_sampling', False)
    if balanced_sampling:
        train_classes = np.argmax(train_labels, axis=1)
        class_counts = np.bincount(train_classes, minlength=len(config['data']['class_names']))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[train_classes]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True
        )
        shuffle = False
        logger.info("WeightedRandomSampler enabled for training DataLoader.")
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
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
    Build train/val/test DataLoaders using INTER-PATIENT splitting.

    This function:
    1. Loads records listed in config['data']['train_records'] → train+val pool
    2. Loads records listed in config['data']['test_records'] → test set
    3. Splits train pool into train/val using val_split_fraction (patient-level)
    4. Processes each group INDEPENDENTLY — no patient beats cross split boundaries
    5. Computes normalization stats from training data ONLY

    Args:
        config (dict): Parsed config.yaml.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
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

    # ── Step 2: Check for cached splits or process independently ──
    processed_dir = os.path.join(os.getcwd(), data_cfg.get('processed_dir', 'software/data/processed'))
    in_channels = data_cfg.get('in_channels', 1)
    ch_suffix = f"_ch{in_channels}" if in_channels > 1 else ""
    cache_path = os.path.join(processed_dir, f"splits_seed{config['project']['seed']}_w{data_cfg['window_size']}{ch_suffix}.npz")

    use_cache = os.path.exists(cache_path)
    if use_cache:
        cached_data = np.load(cache_path)
        if 'train_rr' not in cached_data:
            logger.info(f"Cached file {cache_path} missing RR features. Reprocessing records...")
            use_cache = False

    if use_cache:
        logger.info(f"Loading cached preprocessed splits from {cache_path}...")
        train_windows = cached_data['train_windows']
        train_labels = cached_data['train_labels']
        train_rr = cached_data['train_rr']
        val_windows = cached_data['val_windows']
        val_labels = cached_data['val_labels']
        val_rr = cached_data['val_rr']
        test_windows = cached_data['test_windows']
        test_labels = cached_data['test_labels']
        test_rr = cached_data['test_rr']
        train_mean = float(cached_data['train_mean'])
        train_std = float(cached_data['train_std'])
    else:
        logger.info("Processing training records...")
        train_windows, train_labels, train_rr = _process_records(pure_train_ids, config)

        logger.info("Processing validation records...")
        val_windows, val_labels, val_rr = _process_records(val_record_ids, config)

        logger.info("Processing test records...")
        test_windows, test_labels, test_rr = _process_records(test_record_ids, config)

        # ── Step 3: Compute normalization stats from training data ONLY ──
        train_mean, train_std = compute_normalization_stats(train_windows)

        os.makedirs(processed_dir, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_windows=train_windows, train_labels=train_labels, train_rr=train_rr,
            val_windows=val_windows, val_labels=val_labels, val_rr=val_rr,
            test_windows=test_windows, test_labels=test_labels, test_rr=test_rr,
            train_mean=train_mean, train_std=train_std
        )
        logger.info(f"Cached preprocessed splits to {cache_path}")

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
        train_mean, train_std,
        train_rr=train_rr,
        val_rr=val_rr,
        test_rr=test_rr,
    )
