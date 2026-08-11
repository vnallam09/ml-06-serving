# ============================================================
# tests/test_model_builder_teja.py
# ============================================================
# WHY: Tests for my Phase 5 copy of model_builder_case.py -
# a Titanic survival classifier instead of a penguin species classifier.
#
# Run:
#   uv run pytest tests/test_model_builder_teja.py -v

from pathlib import Path

import joblib
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.ensemble import RandomForestClassifier

import mlstudio.model_builder_teja as mb
from mlstudio.model_builder_teja import (
    FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    evaluate_model,
    load_data,
    split_data,
    train_model,
)

# === TYPE ALIAS ===

SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

# === FIXTURES ===


@pytest.fixture
def df_model() -> pd.DataFrame:
    """Load the model-ready dataframe once for all tests."""
    return load_data()


@pytest.fixture
def split(df_model: pd.DataFrame) -> SplitResult:
    """Split the data once for all tests that need train/test sets."""
    return split_data(df_model)


@pytest.fixture
def model(split: SplitResult) -> RandomForestClassifier:
    """Train the model once for all tests that need a fitted model."""
    X_train, _, y_train, _ = split
    return train_model(X_train, y_train)


# === SECTION 2. LOAD DATA TESTS ===


def test_load_data_returns_dataframe(df_model: pd.DataFrame) -> None:
    """load_data returns a non-empty DataFrame."""
    assert isinstance(df_model, pd.DataFrame)
    assert df_model.shape[0] > 0


def test_load_data_has_required_columns(df_model: pd.DataFrame) -> None:
    """load_data returns a DataFrame with all required columns."""
    required = [TARGET_COL, *FEATURE_COLS]
    for col in required:
        assert col in df_model.columns, f"Missing column: {col}"


def test_load_data_no_missing_values(df_model: pd.DataFrame) -> None:
    """load_data returns a DataFrame with no missing values in required columns."""
    required = [TARGET_COL, *FEATURE_COLS]
    assert df_model[required].isna().sum().sum() == 0


def test_load_data_encodes_sex(df_model: pd.DataFrame) -> None:
    """load_data encodes 'sex' as numeric 0/1, not the original strings."""
    assert set(df_model["sex"].unique()).issubset({0, 1})


# === SECTION 3. SPLIT DATA TESTS ===


def test_split_returns_four_parts(split: SplitResult) -> None:
    """split_data returns exactly four parts."""
    assert len(split) == 4


def test_split_sizes(df_model: pd.DataFrame, split: SplitResult) -> None:
    """Train and test sizes match TEST_SIZE."""
    X_train, X_test, _, _ = split
    total = len(df_model)
    expected_test = round(total * TEST_SIZE)
    assert len(X_test) == pytest.approx(expected_test, abs=1)
    assert len(X_train) + len(X_test) == total


def test_split_feature_columns(split: SplitResult) -> None:
    """X_train and X_test contain exactly FEATURE_COLS."""
    X_train, X_test, _, _ = split
    assert list(X_train.columns) == FEATURE_COLS
    assert list(X_test.columns) == FEATURE_COLS


def test_split_target_values(split: SplitResult) -> None:
    """y_train and y_test contain only known survival labels (0 or 1)."""
    _, _, y_train, y_test = split
    known = {0, 1}
    assert set(y_train.unique()).issubset(known)
    assert set(y_test.unique()).issubset(known)


# === SECTION 4. TRAIN MODEL TESTS ===


def test_train_model_returns_classifier(model: RandomForestClassifier) -> None:
    """train_model returns a fitted RandomForestClassifier."""
    assert isinstance(model, RandomForestClassifier)


def test_train_model_has_feature_names(model: RandomForestClassifier) -> None:
    """The fitted model knows its feature names."""
    # WHY ignore: sklearn sets feature_names_in_ dynamically during fit();
    # it isn't declared on the RandomForestClassifier stub, so pyright
    # doesn't know about it.
    assert list(model.feature_names_in_) == FEATURE_COLS  # pyright: ignore[reportAttributeAccessIssue]


def test_train_model_classes(model: RandomForestClassifier) -> None:
    """The fitted model knows both survival outcomes."""
    assert set(model.classes_) == {0, 1}


# === SECTION 5. EVALUATE MODEL TESTS ===


def test_evaluate_model_runs(model: RandomForestClassifier, split: SplitResult) -> None:
    """evaluate_model runs without raising an exception."""
    _, X_test, _, y_test = split
    evaluate_model(model, X_test, y_test)  # should not raise


def test_evaluate_model_reasonable_accuracy(
    model: RandomForestClassifier, split: SplitResult
) -> None:
    """Test accuracy on titanic should be well above chance (> 0.70)."""
    from sklearn.metrics import accuracy_score

    _, X_test, _, y_test = split
    acc: float = float(accuracy_score(y_test, model.predict(X_test)))
    assert acc > 0.70, f"Accuracy too low: {acc:.3f}"


# === SECTION 6. SAVE MODEL TESTS ===


def test_save_model_creates_file(
    model: RandomForestClassifier,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """save_model writes a file to MODEL_PATH."""
    tmp_model_path = tmp_path / "model_teja.joblib"
    monkeypatch.setattr(mb, "MODEL_PATH", tmp_model_path)

    mb.save_model(model)

    assert tmp_model_path.exists()
    assert tmp_model_path.stat().st_size > 0


def test_save_model_reloadable(
    model: RandomForestClassifier,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """The saved artifact can be reloaded with joblib and still predicts."""
    tmp_model_path = tmp_path / "model_teja.joblib"
    monkeypatch.setattr(mb, "MODEL_PATH", tmp_model_path)

    mb.save_model(model)

    reloaded = joblib.load(tmp_model_path)
    sample = pd.DataFrame(
        [
            {
                "pclass": 1,
                "sex": 1,  # encoded "female"
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
            }
        ]
    )
    prediction = reloaded.predict(sample)
    assert prediction[0] in {0, 1}
