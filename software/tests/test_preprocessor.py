import pytest
import numpy as np
import wfdb
from src.data.preprocessor import (
    detect_r_peaks,
    extract_windows,
    normalize_window,
    extract_labels,
    compute_normalization_stats,
)

# Test 1: test_detect_r_peaks_returns_array
def test_detect_r_peaks_returns_array():
    signal = np.sin(np.linspace(0, 100, 3600))
    # Add sharp spikes every 360 samples
    for i in range(0, 3600, 360):
        signal[i] += 10.0
    result = detect_r_peaks(signal, fs=360)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert len(result) > 0

# Test 2: test_detect_r_peaks_short_signal_raises
def test_detect_r_peaks_short_signal_raises():
    signal = np.zeros(100)
    with pytest.raises(ValueError):
        detect_r_peaks(signal, fs=360)

# Test 3: test_detect_r_peaks_non_1d_raises
def test_detect_r_peaks_non_1d_raises():
    signal = np.zeros((100, 2))
    with pytest.raises(ValueError):
        detect_r_peaks(signal, fs=360)

# Test 4: test_extract_windows_correct_shape
def test_extract_windows_correct_shape():
    signal = np.zeros(3600)
    r_peaks = np.array([360, 720, 1080, 1440, 1800, 2160, 2520, 2880])
    windows, valid_peak_indices = extract_windows(signal, r_peaks, window_size=256)
    assert windows.shape == (8, 256)
    assert valid_peak_indices.shape == (8,)

# Test 5: test_extract_windows_boundary_peaks_skipped
def test_extract_windows_boundary_peaks_skipped():
    signal = np.zeros(3600)
    r_peaks = np.array([10, 720, 3595])
    windows, valid_peak_indices = extract_windows(signal, r_peaks, window_size=256)
    assert windows.shape[0] == 1

# Test 6: test_extract_windows_odd_size_raises
def test_extract_windows_odd_size_raises():
    signal = np.zeros(3600)
    r_peaks = np.array([720])
    with pytest.raises(ValueError):
        extract_windows(signal, r_peaks, window_size=255)

# Test 7: test_normalize_window_zero_mean_unit_std
def test_normalize_window_zero_mean_unit_std():
    window = np.random.randn(256) * 2.0 + 5.0
    result = normalize_window(window)
    assert abs(result.mean()) < 1e-5
    assert abs(result.std() - 1.0) < 1e-4
    assert result.dtype == np.float32

# Test 8: test_normalize_window_flat_signal_no_nan
def test_normalize_window_flat_signal_no_nan():
    window = np.ones(256) * 5.0
    result = normalize_window(window)
    assert not np.any(np.isnan(result))
    assert result.dtype == np.float32

# Test 9: test_extract_labels_correct_class_assignment
def test_extract_labels_correct_class_assignment():
    ann = wfdb.Annotation(
        record_name='test',
        extension='atr',
        sample=np.array([360, 720, 1080]),
        symbol=['N', 'L', 'A'],
        subtype=None, chan=None, num=None, aux_note=None
    )
    valid_peak_indices = np.array([360, 720, 1080])
    annotation_map = {"ST_segment": ["S","J"], "QT_interval": ["f","Q","q"], "P_wave": ["A","a","e","j"], "Bundle_Branch_Block": ["B","L","R","r"], "Normal": ["N","."]}
    class_names = ["ST_segment", "QT_interval", "P_wave", "Bundle_Branch_Block", "Normal"]
    
    labels = extract_labels(ann, valid_peak_indices, annotation_map, class_names)
    assert labels[0, 4] == 1.0
    assert labels[1, 3] == 1.0
    assert labels[2, 2] == 1.0

# Test 10: test_compute_normalization_stats_correct
def test_compute_normalization_stats_correct():
    windows = np.ones((100, 256)) * 3.0
    mean, std = compute_normalization_stats(windows)
    assert abs(mean - 3.0) < 1e-6
    assert abs(std) < 1e-6

# Test 11: test_compute_normalization_stats_empty_raises
def test_compute_normalization_stats_empty_raises():
    windows = np.zeros((0, 256))
    with pytest.raises(ValueError):
        compute_normalization_stats(windows)
