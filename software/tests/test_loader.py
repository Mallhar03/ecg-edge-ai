import pytest
import numpy as np
import wfdb
from pathlib import Path
from src.data.loader import load_record, load_multiple_records

# Test 1: test_load_record_from_disk_signal_shape
def test_load_record_from_disk_signal_shape(tmp_path):
    # In tmp_path, create a synthetic 2-channel WFDB record with 3600 samples
    signal = np.random.randn(3600, 2).astype(np.float32)
    wfdb.wrsamp(
        record_name='100',
        fs=360,
        units=['mV', 'mV'],
        sig_name=['MLII', 'V1'],
        p_signal=signal.astype(np.float64),
        fmt=['16', '16'],
        adc_gain=[200.0, 200.0],
        baseline=[0, 0],
        write_dir=str(tmp_path)
    )
    
    # Create a synthetic annotation file
    wfdb.wrann(
        record_name='100',
        extension='atr',
        sample=np.array([360, 720, 1080]),
        symbol=['N', 'N', 'L'],
        write_dir=str(tmp_path)
    )
    
    # Call load_record('100', str(tmp_path), lead_index=0)
    signal_array, annotation = load_record('100', str(tmp_path), lead_index=0)
    
    # Assertions
    assert signal_array.shape == (3600,)
    assert signal_array.dtype == np.float64
    assert annotation is not None
    assert len(annotation.sample) == 3

# Test 2: test_load_record_lead_index_negative_raises
def test_load_record_lead_index_negative_raises():
    with pytest.raises(ValueError, match="lead_index must be >= 0"):
        load_record("100", "/any/path", lead_index=-1)

# Test 3: test_load_record_empty_id_raises
def test_load_record_empty_id_raises():
    with pytest.raises(ValueError, match="record_id must be a non-empty string"):
        load_record("", "/any/path")

# Test 4: test_load_record_missing_no_streaming_raises
def test_load_record_missing_no_streaming_raises():
    with pytest.raises(FileNotFoundError):
        load_record("999", "/nonexistent/path/for/test", physionet_db=None)

# Test 5: test_load_multiple_records_skip_missing
def test_load_multiple_records_skip_missing(tmp_path):
    # Create a synthetic valid record
    signal = np.random.randn(3600, 1).astype(np.float32)
    wfdb.wrsamp(
        record_name='100',
        fs=360,
        units=['mV'],
        sig_name=['MLII'],
        p_signal=signal.astype(np.float64),
        fmt=['16'],
        adc_gain=[200.0],
        baseline=[0],
        write_dir=str(tmp_path)
    )
    wfdb.wrann(
        record_name='100',
        extension='atr',
        sample=np.array([100]),
        symbol=['N'],
        write_dir=str(tmp_path)
    )
    
    # Call load_multiple_records(["100", "nonexistent_xyz"], str(tmp_path), skip_missing=True)
    result = load_multiple_records(["100", "nonexistent_xyz"], str(tmp_path), skip_missing=True)
    
    # Assertions
    assert len(result) == 1
    assert "100" in result
    assert "nonexistent_xyz" not in result

# Test 6: test_load_multiple_records_empty_list_raises
def test_load_multiple_records_empty_list_raises():
    with pytest.raises(ValueError, match="record_ids list is empty"):
        load_multiple_records([], "/any/path")

# Test 7: test_load_multiple_records_no_skip_raises_on_missing
def test_load_multiple_records_no_skip_raises_on_missing():
    with pytest.raises(FileNotFoundError):
        load_multiple_records(["nonexistent_xyz"], "/nonexistent/path", skip_missing=False)

# Dataset Tests appended for Phase 1A
from src.data.dataset import ECGDataset
import torch

# Test: test_ecg_dataset_getitem_shape
def test_ecg_dataset_getitem_shape():
    windows = np.random.randn(10, 256).astype(np.float64)
    labels = np.zeros((10, 5), dtype=np.float32)
    labels[:, 4] = 1.0
    dataset = ECGDataset(windows, labels, mean=0.0, std=1.0)
    assert len(dataset) == 10
    x, y = dataset[0]
    assert x.shape == torch.Size([1, 256])
    assert y.shape == torch.Size([5])
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

# Test: test_ecg_dataset_label_count_mismatch_raises
def test_ecg_dataset_label_count_mismatch_raises():
    windows = np.random.randn(10, 256)
    labels = np.zeros((9, 5))
    with pytest.raises(ValueError):
        ECGDataset(windows, labels)
