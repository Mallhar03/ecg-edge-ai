import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
import wfdb
import wfdb.processing

logger = logging.getLogger(__name__)

def detect_r_peaks(
    signal: np.ndarray,
    fs: int,
    min_hr_bpm: int = 30,
    max_hr_bpm: int = 300
) -> np.ndarray:
    """
    Detect R-peak locations in a 1D ECG signal using the XQRS algorithm.

    Uses wfdb.processing.xqrs_detect and applies physiological plausibility
    filtering to remove spurious detections outside [min_hr_bpm, max_hr_bpm].

    Args:
        signal: 1D numpy array of ECG samples, dtype float.
        fs: Sampling frequency in Hz. For MIT-BIH, always 360.
        min_hr_bpm: Minimum physiologically valid heart rate in BPM.
        max_hr_bpm: Maximum physiologically valid heart rate in BPM.

    Returns:
        np.ndarray of shape (K,), dtype int64. Sample indices of detected
        R-peaks sorted ascending. Returns empty array if none detected.

    Raises:
        ValueError: If signal is not 1D, too short for detection, or fs <= 0.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if len(signal) < 2 * fs:
        raise ValueError(f"Signal too short ({len(signal)} samples) for reliable R-peak detection at {fs} Hz. Minimum: {2*fs} samples.")

    raw_peaks = wfdb.processing.xqrs_detect(sig=signal, fs=fs, verbose=False)
    
    if len(raw_peaks) == 0:
        logger.warning(f"No R-peaks detected in signal of length {len(signal)}. Check signal quality and lead index.")
        return np.array([], dtype=np.int64)

    min_samples = int(60 / max_hr_bpm * fs)
    max_samples = int(60 / min_hr_bpm * fs)
    
    diffs = np.diff(raw_peaks)
    valid_gaps = (diffs >= min_samples) & (diffs <= max_samples)
    
    keep = np.zeros(len(raw_peaks), dtype=bool)
    if len(raw_peaks) > 1:
        keep[:-1] |= valid_gaps
        keep[1:] |= valid_gaps
    
    r_peaks = raw_peaks[keep].astype(np.int64)
    
    if len(r_peaks) == 0:
        logger.warning(f"No R-peaks detected in signal of length {len(signal)}. Check signal quality and lead index.")
        return np.array([], dtype=np.int64)

    mean_hr = 0.0
    final_diffs = np.diff(r_peaks)
    if len(final_diffs) > 0:
        mean_hr = 60.0 / (np.mean(final_diffs) / fs)

    logger.info(f"Detected {len(r_peaks)} R-peaks in {len(signal)/fs:.1f}s signal. Mean HR: {mean_hr:.1f} bpm.")
    return r_peaks

def extract_windows(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-size signal windows centered on R-peak locations.

    The R-peak is placed at index window_size//2 within each window.
    R-peaks too close to signal boundaries are silently skipped — never errored.
    This is the correct behavior: boundary beats simply have insufficient context.

    Args:
        signal: 1D ECG signal array of shape (N,).
        r_peaks: Array of R-peak sample indices, shape (K,), from detect_r_peaks().
        window_size: Number of samples per window. Must be positive and even.
                     256 at 360 Hz = 711ms, capturing full P-QRS-T morphology.

    Returns:
        Tuple of:
            windows: np.ndarray of shape (M, window_size), dtype float64.
                     M <= K because boundary R-peaks are dropped.
            valid_peak_indices: np.ndarray of shape (M,), dtype int64.
                                R-peak sample indices for each extracted window.

    Raises:
        ValueError: If signal is not 1D, window_size is not a positive even
                    integer, or r_peaks is not 1D.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got ndim={signal.ndim}")
    if window_size <= 0 or window_size % 2 != 0:
        raise ValueError(f"window_size must be a positive even integer, got {window_size}")
    if r_peaks.ndim != 1:
        raise ValueError(f"r_peaks must be 1D, got ndim={r_peaks.ndim}")

    half = window_size // 2
    windows = []
    valid_peak_indices = []

    for p in r_peaks:
        if p - half < 0 or p + half > len(signal):
            logger.debug(f"Skipping R-peak at sample {p}: boundary violation (signal_len={len(signal)}, half={half})")
            continue
        windows.append(signal[p - half : p + half])
        valid_peak_indices.append(p)

    logger.info(f"Extracted {len(windows)}/{len(r_peaks)} windows of size {window_size}. Skipped {len(r_peaks)-len(windows)} boundary R-peaks.")
    
    if len(windows) == 0:
        return np.empty((0, window_size), dtype=np.float64), np.empty((0,), dtype=np.int64)
        
    return np.array(windows, dtype=np.float64), np.array(valid_peak_indices, dtype=np.int64)

def normalize_window(
    window: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """
    Normalize a single ECG window to zero mean and unit variance.

    If mean/std are not provided, computes statistics from the window itself
    (per-window normalization). For training/inference, pass pre-computed
    training-set statistics to prevent data leakage into val/test sets.

    A small epsilon (1e-8) prevents division by zero for flat-line signals.

    Args:
        window: 1D array of ECG samples, shape (window_size,).
        mean: Pre-computed mean. If None, computed from the window.
        std: Pre-computed standard deviation. If None, computed from the window.

    Returns:
        Normalized window as np.ndarray of shape (window_size,), dtype float32.

    Raises:
        ValueError: If window is not 1D, is empty, or std < 0.
    """
    if window.ndim != 1:
        raise ValueError(f"window must be 1D, got ndim={window.ndim}")
    if len(window) == 0:
        raise ValueError("window must not be empty")
    if std is not None and std < 0:
        raise ValueError(f"std must be non-negative, got {std}")

    m = window.mean() if mean is None else mean
    s = window.std() if std is None else std

    return ((window - m) / (s + 1e-8)).astype(np.float32)


def _build_symbol_to_class_map(
    annotation_map: Dict[str, List[str]],
    class_names: List[str]
) -> Dict[str, int]:
    """
    Build a flat lookup table: annotation_symbol -> class_index.

    This inverts the annotation_map (class_name -> [symbols]) into a
    direct symbol -> class_index dictionary for O(1) lookup per beat.

    Every symbol that appears in annotation_map gets exactly one class.
    Symbols NOT in this map will be logged as unmapped and EXCLUDED
    (not silently labeled as Normal).

    Args:
        annotation_map: Dict mapping class name -> list of annotation symbols.
        class_names: Ordered list of class names defining index positions.

    Returns:
        Dict mapping each annotation symbol (str) to its class index (int).

    Raises:
        ValueError: If a symbol appears in multiple classes.
    """
    sym_to_class = {}
    for class_idx, name in enumerate(class_names):
        symbols = annotation_map.get(name, [])
        for sym in symbols:
            if sym in sym_to_class:
                raise ValueError(
                    f"Symbol '{sym}' is mapped to both class '{class_names[sym_to_class[sym]]}' "
                    f"and class '{name}'. Each symbol must belong to exactly one class."
                )
            sym_to_class[sym] = class_idx
    return sym_to_class


def extract_labels(
    annotation: wfdb.Annotation,
    valid_peak_indices: np.ndarray,
    annotation_map: Dict[str, List[str]],
    class_names: List[str],
    search_radius: int = 12
) -> np.ndarray:
    """
    Extract multi-hot class labels for each R-peak from WFDB annotations.

    Uses np.searchsorted for O(log M) annotation lookup per R-peak (BUG 4 fix),
    then finds the CLOSEST annotation within search_radius (not just the first hit).

    Beats whose annotation symbol is NOT found in ANY class are EXCLUDED
    (labeled all-zeros). The caller must filter these out. This prevents the
    critical BUG 2 where unmapped symbols (V, /, F) were silently labeled Normal.

    Args:
        annotation: wfdb.Annotation object from wfdb.rdann().
        valid_peak_indices: 1D array of R-peak sample indices, shape (M,).
        annotation_map: Dict mapping class name -> list of annotation symbols.
                        e.g. {"Ventricular": ["V", "F"], "Normal": ["N", "."]}
        class_names: Ordered list of class names. Order defines label positions.
        search_radius: Samples to search around each R-peak for annotation match.
                        Default 12 accounts for XQRS jitter at 360 Hz (~33ms).

    Returns:
        np.ndarray of shape (M, num_classes), dtype float32. Multi-hot encoded labels.
        Row i corresponds to valid_peak_indices[i].
        Rows with all zeros indicate unmapped beats.

    Raises:
        ValueError: If valid_peak_indices is not 1D.
    """
    if valid_peak_indices.ndim != 1:
        raise ValueError(f"valid_peak_indices must be 1D, got ndim={valid_peak_indices.ndim}")

    num_classes = len(class_names)
    labels = np.zeros((len(valid_peak_indices), num_classes), dtype=np.float32)

    # Build O(1) symbol -> class_index lookup
    sym_to_class = _build_symbol_to_class_map(annotation_map, class_names)

    # Sort annotation samples for binary search (should already be sorted, but be safe)
    ann_samples = np.array(annotation.sample, dtype=np.int64)
    ann_symbols = annotation.symbol

    # Track unmapped symbols for logging
    unmapped_counts = {}
    mapped_count = 0
    no_match_count = 0

    for i, p in enumerate(valid_peak_indices):
        # O(log M) binary search: find the insertion point
        insert_idx = np.searchsorted(ann_samples, p)

        # Check candidates in the neighborhood around insert_idx
        best_idx = -1
        best_dist = search_radius + 1

        for candidate_idx in range(max(0, insert_idx - 2), min(len(ann_samples), insert_idx + 3)):
            dist = abs(int(ann_samples[candidate_idx]) - int(p))
            if dist <= search_radius and dist < best_dist:
                best_dist = dist
                best_idx = candidate_idx

        if best_idx != -1:
            sym = ann_symbols[best_idx]
            if sym in sym_to_class:
                labels[i, sym_to_class[sym]] = 1.0
                mapped_count += 1
            else:
                # NOT in any class — leave as all-zeros (will be filtered)
                unmapped_counts[sym] = unmapped_counts.get(sym, 0) + 1
        else:
            no_match_count += 1

    # Log comprehensive statistics
    label_counts = labels.sum(axis=0).astype(int)
    logger.info(
        f"Extracted labels for {len(valid_peak_indices)} beats. "
        f"Mapped: {mapped_count}, No annotation match: {no_match_count}. "
        f"Class distribution: {dict(zip(class_names, label_counts.tolist()))}"
    )
    if unmapped_counts:
        logger.warning(
            f"Unmapped annotation symbols (excluded, NOT defaulted to Normal): {unmapped_counts}. "
            f"Total excluded: {sum(unmapped_counts.values())} beats."
        )

    return labels


def compute_normalization_stats(windows: np.ndarray) -> Tuple[float, float]:
    """
    Compute global normalization statistics over the training set windows.

    CRITICAL: Call this ONLY on training set windows. Applying the returned
    statistics uniformly to val/test sets is correct and required to prevent
    data leakage. Never call this on val or test windows.

    Args:
        windows: np.ndarray of shape (N, window_size). Training windows only.

    Returns:
        Tuple of (mean, std) as Python floats.
        Suitable for JSON serialization and reuse in normalize_window().

    Raises:
        ValueError: If windows is not 2D or has 0 rows.
    """
    if windows.ndim != 2:
        raise ValueError(f"windows must be 2D (N, window_size), got ndim={windows.ndim}")
    if windows.shape[0] == 0:
        raise ValueError("windows array is empty — cannot compute normalization stats")

    mean = float(np.mean(windows))
    std = float(np.std(windows))

    logger.info(f"Normalization stats computed on {windows.shape[0]} training windows. Mean: {mean:.6f}, Std: {std:.6f}")
    return mean, std
