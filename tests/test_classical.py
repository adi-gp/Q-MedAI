from src.config import RANDOM_SEED
from src.data.loader import load_dataset
from src.models.classical import CLASSICAL_MODEL_NAMES, build_classical_model
from src.preprocessing.pipeline import split_and_preprocess


def test_all_classical_models_train_and_produce_continuous_scores():
    bundle = load_dataset()
    prepared = split_and_preprocess(bundle.X, bundle.y, n_features=4)
    for name in CLASSICAL_MODEL_NAMES:
        model = build_classical_model(name, RANDOM_SEED)
        model.fit(prepared.X_train, prepared.y_train)
        scores = model.predict_proba(prepared.X_test)[:, 1]
        assert scores.shape == prepared.y_test.shape
        assert ((scores >= 0) & (scores <= 1)).all()
