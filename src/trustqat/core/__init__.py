"""Framework-free building blocks (numpy only)."""

from .compare import paired_summary, relative_reduction
from .metrics import (
    brier_score,
    calibration_bins,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    softmax,
)
from .pareto import ParetoArchive, pareto_front, select_records
from .records import EpochRecord, read_records, validate_record, write_record
from .temperature import apply_temperature, fit_temperature

__all__ = [
    'EpochRecord',
    'ParetoArchive',
    'apply_temperature',
    'brier_score',
    'calibration_bins',
    'calibration_report',
    'expected_calibration_error',
    'fit_temperature',
    'maximum_calibration_error',
    'negative_log_likelihood',
    'paired_summary',
    'pareto_front',
    'read_records',
    'relative_reduction',
    'select_records',
    'softmax',
    'validate_record',
    'write_record',
]
