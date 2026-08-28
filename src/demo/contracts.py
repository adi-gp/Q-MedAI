from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class DemoError(RuntimeError):
    """Base error for the combined faculty demo."""


class EvidenceError(DemoError):
    """Evidence is missing or violates the expected schema."""


class ArtifactError(DemoError):
    """Frozen model artifacts cannot be trusted or used."""


class PatientValidationError(DemoError):
    """Patient input does not satisfy the frozen model contract."""


@dataclass(frozen=True)
class MetricRecord:
    model: str
    family: str
    accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    roc_auc: float
    auprc: float | None = None
    precision: float | None = None
    false_negatives: int | None = None
    training_seconds: float | None = None


@dataclass(frozen=True)
class TaskEvidence:
    task_id: str
    title: str
    role: str
    task_type: str
    source: str
    cohort: Mapping[str, Any]
    comparison_fairness: str
    backend: str
    metrics: tuple[MetricRecord, ...]
    focus_reference_model: str
    focus_candidate_model: str
    prioritized_metric: str
    uncertainty_status: str
    calibration_status: str
    limitations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UtilityVerdict:
    task_id: str
    status: str
    headline: str
    reference_model: str
    candidate_model: str
    deltas: Mapping[str, float]
    best_overall_model: str
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactStatus:
    ready: bool
    code: str
    message: str
    manifest: Mapping[str, Any] | None = None
    missing_files: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FraminghamArtifacts:
    manifest: Mapping[str, Any]
    classical: Mapping[str, Any]
    quantum: Mapping[str, Any]
    hybrid: Mapping[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    classical_score: float
    quantum_score: float
    hybrid_score: float
    recommended_pathway: str
    contributions: tuple[tuple[str, float], ...]
