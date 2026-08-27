"""Shared experiment defaults and reproducibility settings."""

from dataclasses import dataclass


RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.20
# Final SIH experiment defaults. The verified 2-layer/35-iteration/50-row
# configuration remains documented as the local baseline; it is not rerun.
DEFAULT_VQC_ITERATIONS = 100
DEFAULT_VQC_LAYERS = 3
DEFAULT_QUANTUM_TIMEOUT_SECONDS = 600
DEFAULT_KERNEL_SUBSET_SIZE = 150
MAX_KERNEL_SUBSET_SIZE = 150
SUPPORTED_FEATURE_COUNTS = (4, 6, 8)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one reproducible, local experiment."""

    reduction_method: str = "SelectKBest"
    n_features: int = 4
    test_size: float = DEFAULT_TEST_SIZE
    vqc_layers: int = DEFAULT_VQC_LAYERS
    vqc_iterations: int = DEFAULT_VQC_ITERATIONS
    kernel_subset_size: int = DEFAULT_KERNEL_SUBSET_SIZE
    quantum_timeout_seconds: int = DEFAULT_QUANTUM_TIMEOUT_SECONDS
    random_seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.reduction_method not in {"PCA", "SelectKBest"}:
            raise ValueError("reduction_method must be 'PCA' or 'SelectKBest'.")
        if self.n_features not in SUPPORTED_FEATURE_COUNTS:
            raise ValueError(f"n_features must be one of {SUPPORTED_FEATURE_COUNTS}.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.vqc_layers < 1 or self.vqc_iterations < 1:
            raise ValueError("VQC layers and iterations must be positive.")
        if not 2 <= self.kernel_subset_size <= MAX_KERNEL_SUBSET_SIZE:
            raise ValueError(
                f"kernel_subset_size must be between 2 and {MAX_KERNEL_SUBSET_SIZE}."
            )
