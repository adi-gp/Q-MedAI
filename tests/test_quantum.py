import numpy as np
import pytest

from src.models.quantum import (
    QuantumKernelSVM,
    QuantumTimeoutError,
    VQCClassifier,
    select_stratified_subset,
)


@pytest.fixture
def small_binary_data():
    X = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.1, 0.3, -0.2, 0.5],
            [0.2, -0.3, 0.1, -0.4],
            [-0.2, -0.1, 0.4, 0.2],
            [0.3, 0.0, -0.3, 0.1],
            [-0.3, 0.2, 0.2, -0.2],
        ]
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    return X, y


def test_vqc_circuit_executes_and_predicts_one_score_per_row(small_binary_data):
    X, y = small_binary_data
    model = VQCClassifier(n_qubits=4, layers=1, iterations=1, timeout_seconds=30)
    model.fit(X, y)
    scores = model.predict_scores(X)
    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_quantum_kernel_train_and_test_matrix_shapes(small_binary_data):
    X, y = small_binary_data
    model = QuantumKernelSVM(n_qubits=4, subset_size=4, timeout_seconds=30)
    model.fit(X, y)
    train_kernel = model.kernel_matrix(model.X_train_subset, model.X_train_subset, symmetric=True)
    test_kernel = model.kernel_matrix(X[:2], model.X_train_subset)
    assert train_kernel.shape == (4, 4)
    assert test_kernel.shape == (2, 4)
    assert np.allclose(train_kernel, train_kernel.T)


def test_stratified_subsampling_preserves_approximate_class_ratio():
    X = np.arange(400, dtype=float).reshape(100, 4)
    y = np.array([0] * 70 + [1] * 30)
    _, subset_y, info = select_stratified_subset(X, y, requested_size=50)
    assert info.original_training_size == 100
    assert info.selected_subset_size == 50
    assert abs(subset_y.mean() - y.mean()) <= 0.03


def test_vqc_cooperative_timeout_raises_without_results(small_binary_data):
    X, y = small_binary_data
    model = VQCClassifier(n_qubits=4, layers=1, iterations=1, timeout_seconds=0)
    with pytest.raises(QuantumTimeoutError):
        model.fit(X, y)
    assert model.weights is None
