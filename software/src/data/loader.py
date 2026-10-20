import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import wfdb

logger = logging.getLogger(__name__)

def load_record(
    record_id: str,
    data_dir: str,
    lead_index: Union[int, List[int]] = 0,
    physionet_db: Optional[str] = None
) -> Tuple[np.ndarray, wfdb.Annotation]:
    """
    Load a single MIT-BIH WFDB record and its beat annotations.

    Attempts disk load first. Falls back to PhysioNet streaming only if
    physionet_db is provided and the file is not found locally.

    Args:
        record_id: MIT-BIH record identifier, e.g. "100", "201".
        data_dir: Directory containing .hea/.dat/.atr files.
        lead_index: ECG lead channel index (e.g. 0 for MLII, or [0, 1] for dual-lead).
        physionet_db: PhysioNet database shortname for streaming fallback.

    Returns:
        Tuple of:
            signal_array: np.ndarray of shape (N,) or (N, C), dtype float64.
            annotation: wfdb.Annotation with .sample (indices) and .symbol (beat codes).
    """
    # Input validation
    if not record_id or not isinstance(record_id, str):
        raise ValueError("record_id must be a non-empty string")
    if isinstance(lead_index, int) and lead_index < 0:
        raise ValueError(f"lead_index must be >= 0, got {lead_index}")

    record_path = Path(data_dir) / record_id
    found_on_disk = (Path(data_dir) / f"{record_id}.hea").exists()

    try:
        if found_on_disk:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), 'atr')
        elif physionet_db is not None:
            record = wfdb.rdrecord(record_id, pn_dir=physionet_db)
            annotation = wfdb.rdann(record_id, 'atr', pn_dir=physionet_db)
        else:
            logger.error(f"Record {record_id} not found in {data_dir} and no physionet_db provided.")
            raise FileNotFoundError(
                f"Record {record_id} not found. Provide data_dir with .hea/.dat files "
                f"or set physionet_db='{physionet_db or 'mitdb'}' for streaming."
            )

        # Extract signal (1D or multi-channel)
        if isinstance(lead_index, (list, tuple)):
            signal_array = record.p_signal[:, list(lead_index)].astype(np.float64)
        else:
            signal_array = record.p_signal[:, lead_index].astype(np.float64)

        if len(signal_array) == 0:
            raise ValueError(f"Record {record_id} contains no signal samples.")
        if len(signal_array) < 1000:
            logger.warning(
                f"Record {record_id} is suspiciously short: {len(signal_array)} samples. "
                "Check file integrity."
            )

        logger.info(
            f"Loaded record {record_id} from {'disk' if found_on_disk else 'PhysioNet'}. "
            f"Signal length: {len(signal_array)} samples ({len(signal_array)/360:.1f}s). "
            f"Annotations: {len(annotation.sample)}."
        )

        return signal_array, annotation

    except FileNotFoundError:
        # Re-raise explicit FileNotFoundError
        raise
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to load record {record_id}: {str(e)}")

def load_multiple_records(
    record_ids: list,
    data_dir: str,
    lead_index: int = 0,
    physionet_db: Optional[str] = None,
    skip_missing: bool = False
) -> Dict[str, Tuple[np.ndarray, wfdb.Annotation]]:
    """
    Load multiple MIT-BIH records, with optional skip-on-missing behavior.

    Args:
        record_ids: List of record ID strings to load.
        data_dir: Directory containing WFDB files.
        lead_index: ECG lead index. 0 = MLII for MIT-BIH.
        physionet_db: PhysioNet DB name for streaming fallback. None = local only.
        skip_missing: If True, log warning and skip records that fail to load.
                      If False (default), raise on first failure.

    Returns:
        Dict mapping record_id (str) -> (signal_array, annotation) tuple.
        Missing records are absent from dict if skip_missing=True.

    Raises:
        ValueError: If record_ids is not a list or is empty.
        FileNotFoundError: If skip_missing=False and any record is missing.
    """
    # Input validation
    if not isinstance(record_ids, list):
        raise ValueError(f"record_ids must be a list, got {type(record_ids).__name__}")
    if len(record_ids) == 0:
        raise ValueError("record_ids list is empty")

    results = {}
    for rid in record_ids:
        try:
            results[rid] = load_record(rid, data_dir, lead_index, physionet_db)
        except (FileNotFoundError, RuntimeError) as e:
            if skip_missing:
                logger.warning(f"Skipping record {rid}: not found or load error. Details: {str(e)}")
                continue
            else:
                raise

    logger.info(f"Loaded {len(results)}/{len(record_ids)} records successfully.")
    return results
