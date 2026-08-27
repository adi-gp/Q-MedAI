"""Small PennyLane quantum-circuit models run on a classical simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

from src.config import MAX_KERNEL_SUBSET_SIZE, RANDOM_SEED


class QuantumTimeoutError(TimeoutError):
    """A cooperative timeout, checked between individual quantum computations."""


def _angles(values: np.ndarray) -> np.ndarray:
    """Map arbitrary standardized values to valid, bounded RY rotation angles."""
    return np.tanh(np.asarray(values, dtype=float)) * np.pi


def _ensure_before_timeout(start_time: float, timeout_seconds: float) -> None:
    if time.perf_counter() - start_time >= timeout_seconds:
        raise QuantumTimeoutError(
            f"Cooperative quantum timeout exceeded after {timeout_seconds:.0f} seconds."
        )


class VQCClassifier:
    """A compact variational quantum classifier on PennyLane ``default.qubit``.

    ``default.qubit`` is a classical simulator. It does not execute a quantum
    circuit on physical quantum hardware. PennyLane does not expose a universal
    per-operation seed for this analytic simulator, so reproducibility comes from
    NumPy's seeded deterministic weight initialization and optimizer path.
    """

    backend = "PennyLane default.qubit (classical simulation)"

    def __init__(
        self,
        n_qubits: int = 4,
        layers: int = 2,
        iterations: int = 35,
        learning_rate: float = 0.12,
        timeout_seconds: float = 180,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if n_qubits < 1 or layers < 1 or iterations < 1:
            raise ValueError("n_qubits, layers, and iterations must all be positive.")
        self.n_qubits = n_qubits
        self.layers = layers
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.timeout_seconds = timeout_seconds
        self.random_seed = random_seed
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.weights: pnp.ndarray | None = None
        self.loss_history: list[float] = []
        self.training_seconds: float | None = None
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(inputs, weights):
            for wire in range(self.n_qubits):
                qml.RY(inputs[wire], wires=wire)
            for layer in range(self.layers):
                for wire in range(self.n_qubits):
                    qml.RX(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                    qml.RZ(weights[layer, wire, 2], wires=wire)
                if self.n_qubits > 1:
                    for wire in range(self.n_qubits):
                        qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def _check_features(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected a 2D matrix with {self.n_qubits} feature(s), got {values.shape}."
            )
        return _angles(values)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCClassifier":
        X_angles = self._check_features(X)
        labels = pnp.array(np.asarray(y, dtype=float), requires_grad=False)
        if len(X_angles) != len(labels):
            raise ValueError("X and y must have the same number of rows.")
        start_time = time.perf_counter()
        self.training_seconds = None
        np.random.seed(self.random_seed)
        rng = np.random.default_rng(self.random_seed)
        weights = pnp.array(
            rng.normal(loc=0.0, scale=0.05, size=(self.layers, self.n_qubits, 3)),
            requires_grad=True,
        )
        X_train = pnp.array(X_angles, requires_grad=False)

        def loss(current_weights):
            expectations = pnp.stack([self._circuit(row, current_weights) for row in X_train])
            probabilities = pnp.clip((1.0 + expectations) / 2.0, 1e-7, 1.0 - 1e-7)
            return -pnp.mean(
                labels * pnp.log(probabilities) + (1.0 - labels) * pnp.log(1.0 - probabilities)
            )

        optimizer = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        try:
            for _ in range(self.iterations):
                _ensure_before_timeout(start_time, self.timeout_seconds)
                weights, current_loss = optimizer.step_and_cost(loss, weights)
                self.loss_history.append(float(current_loss))
        except Exception:
            self.training_seconds = time.perf_counter() - start_time
            raise
        self.weights = weights
        self.training_seconds = time.perf_counter() - start_time
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("VQC must be fitted before prediction.")
        X_angles = self._check_features(X)
        return np.asarray(
            [(1.0 + float(self._circuit(row, self.weights))) / 2.0 for row in X_angles],
            dtype=float,
        )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classify with the documented probability-like score threshold of 0.5."""
        return (self.predict_scores(X) >= threshold).astype(int)


@dataclass
class KernelSubsetInfo:
    original_training_size: int
    selected_subset_size: int
    original_class_distribution: dict[str, int]
    subset_class_distribution: dict[str, int]


def select_stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    requested_size: int,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, KernelSubsetInfo]:
    """Select a reproducible, stratified training subset for kernel calculation."""
    values = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(values) != len(labels):
        raise ValueError("X and y must have the same length.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Quantum kernel subsampling requires both binary classes.")
    subset_indices = stratified_subset_indices(labels, requested_size, random_seed)

    subset_y = labels[subset_indices]
    distribution = lambda array: {
        str(label): int(count)
        for label, count in zip(*np.unique(array, return_counts=True))
    }
    info = KernelSubsetInfo(
        original_training_size=len(labels),
        selected_subset_size=len(subset_indices),
        original_class_distribution=distribution(labels),
        subset_class_distribution=distribution(subset_y),
    )
    return values[subset_indices], subset_y, info


def stratified_subset_indices(
    y: np.ndarray,
    requested_size: int,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Return reproducible row indices for a stratified subset without fitting data transforms."""
    labels = np.asarray(y, dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Quantum kernel subsampling requires both binary classes.")
    selected_size = min(requested_size, len(labels))
    if selected_size < len(np.unique(labels)):
        raise ValueError("Quantum kernel subset must contain at least one row per class.")
    if selected_size == len(labels):
        return np.arange(len(labels))
    placeholder = np.zeros((len(labels), 1), dtype=float)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=selected_size,
        random_state=random_seed,
    )
    subset_indices, _ = next(splitter.split(placeholder, labels))
    return subset_indices


class QuantumKernelSVM:
    """Fidelity quantum-kernel SVM using train×train and test×train matrices."""

    backend = "PennyLane default.qubit (classical simulation)"

    def __init__(
        self,
        n_qubits: int = 4,
        subset_size: int = 50,
        timeout_seconds: float = 180,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive.")
        if not 2 <= subset_size <= MAX_KERNEL_SUBSET_SIZE:
            raise ValueError(f"subset_size must be between 2 and {MAX_KERNEL_SUBSET_SIZE}.")
        self.n_qubits = n_qubits
        self.subset_size = subset_size
        self.timeout_seconds = timeout_seconds
        self.random_seed = random_seed
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.svc: SVC | None = None
        self.X_train_subset: np.ndarray | None = None
        self.subset_info: KernelSubsetInfo | None = None
        self.training_seconds: float | None = None
        self.train_kernel_seconds: float | None = None
        self.svm_training_seconds: float | None = None
        self.test_kernel_seconds: float | None = None
        self._build_kernel_circuit()

    def _build_kernel_circuit(self) -> None:
        def feature_map(inputs):
            for wire in range(self.n_qubits):
                qml.RY(inputs[wire], wires=wire)

        @qml.qnode(self.device)
        def kernel_circuit(left, right):
            feature_map(left)
            qml.adjoint(feature_map)(right)
            return qml.probs(wires=range(self.n_qubits))

        self._kernel_circuit = kernel_circuit

    def _check_features(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected a 2D matrix with {self.n_qubits} feature(s), got {values.shape}."
            )
        return _angles(values)

    def _fidelity(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(self._kernel_circuit(left, right)[0])

    def kernel_matrix(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        symmetric: bool = False,
        start_time: float | None = None,
    ) -> np.ndarray:
        """Build a fidelity matrix while yielding timeout checks between pair computations."""
        left_angles = self._check_features(left)
        right_angles = self._check_features(right)
        began = time.perf_counter() if start_time is None else start_time
        matrix = np.empty((len(left_angles), len(right_angles)), dtype=float)
        if symmetric and len(left_angles) == len(right_angles) and np.array_equal(left_angles, right_angles):
            for row in range(len(left_angles)):
                _ensure_before_timeout(began, self.timeout_seconds)
                for column in range(row, len(right_angles)):
                    _ensure_before_timeout(began, self.timeout_seconds)
                    value = self._fidelity(left_angles[row], right_angles[column])
                    matrix[row, column] = value
                    matrix[column, row] = value
            return matrix

        for row in range(len(left_angles)):
            _ensure_before_timeout(began, self.timeout_seconds)
            for column in range(len(right_angles)):
                _ensure_before_timeout(began, self.timeout_seconds)
                matrix[row, column] = self._fidelity(left_angles[row], right_angles[column])
        return matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVM":
        start_time = time.perf_counter()
        self.training_seconds = None
        self.train_kernel_seconds = None
        self.svm_training_seconds = None
        try:
            subset_X, subset_y, info = select_stratified_subset(
                X, y, self.subset_size, self.random_seed
            )
            kernel_started = time.perf_counter()
            train_kernel = self.kernel_matrix(subset_X, subset_X, symmetric=True, start_time=start_time)
            self.train_kernel_seconds = time.perf_counter() - kernel_started
            self.svc = SVC(kernel="precomputed", random_state=self.random_seed)
            svm_started = time.perf_counter()
            self.svc.fit(train_kernel, subset_y)
            self.svm_training_seconds = time.perf_counter() - svm_started
            self.X_train_subset = subset_X
            self.subset_info = info
        except Exception:
            self.training_seconds = time.perf_counter() - start_time
            raise
        self.training_seconds = time.perf_counter() - start_time
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate only test×train kernel rows; no test×test kernel is constructed."""
        if self.svc is None or self.X_train_subset is None:
            raise RuntimeError("Quantum kernel SVM must be fitted before prediction.")
        test_train_kernel = self.kernel_matrix(X, self.X_train_subset, symmetric=False)
        return np.asarray(self.svc.decision_function(test_train_kernel), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.svc is None or self.X_train_subset is None:
            raise RuntimeError("Quantum kernel SVM must be fitted before prediction.")
        test_train_kernel = self.kernel_matrix(X, self.X_train_subset, symmetric=False)
        return np.asarray(self.svc.predict(test_train_kernel), dtype=int)

    def predict_with_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct test×train once, then obtain both labels and decision scores."""
        if self.svc is None or self.X_train_subset is None:
            raise RuntimeError("Quantum kernel SVM must be fitted before prediction.")
        started = time.perf_counter()
        test_train_kernel = self.kernel_matrix(X, self.X_train_subset, symmetric=False)
        self.test_kernel_seconds = time.perf_counter() - started
        predictions = np.asarray(self.svc.predict(test_train_kernel), dtype=int)
        scores = np.asarray(self.svc.decision_function(test_train_kernel), dtype=float)
        return predictions, scores
