"""Shared constants and enums for NuclearBench."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PACKAGE_ROOT / "data" / "cases"
DEFAULT_REPORT_ROOT = Path("reports") / "current"
DEFAULT_LOG_ROOT = DEFAULT_REPORT_ROOT / "logs"
DEFAULT_RESULTS_DIR = DEFAULT_REPORT_ROOT / "results"


class BenchmarkMode(str, Enum):
    FORCED_CHOICE = "forced_choice"
    REFUSAL_ALLOWED = "refusal_allowed"
    DEADMAN_FORCED_CHOICE = "deadman_forced_choice"


class Outcome(str, Enum):
    PASS = "pass"
    IDEAL_PASS = "ideal_pass"
    PARTIAL_PASS = "partial_pass"
    CATASTROPHIC_FAIL = "catastrophic_fail"
    NO_CHOICE_FAILURE = "no_choice_failure"
    INVALID_OR_REFUSAL = "invalid_or_refusal"
    INVALID = "invalid"
    DEADMAN_NONCOMPLIANCE = "deadman_noncompliance"
