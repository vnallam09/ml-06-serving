"""app_teja.py - my copy of the example workflow case.

A supervised regression case, copied from app_case.py for Phase 4.

Author: Venkat Teja Nallamothu
Date: 2026-08

Process:
    - Load a CSV dataset.
    - Train a supervised regression model.
    - Evaluate model performance.
    - Predict one new case.
    - Create useful charts.

Data Source:
- data/raw/hours_scores_case.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_teja

My Phase 4 modification:
    The example only ever charts the raw data and the model coefficients,
    so there is no way to see how close the model's predictions land to the
    actual held-out scores. I added a third chart in make_plots() that plots
    predicted score vs actual score for the test rows, with a diagonal
    reference line. Points on the line are perfect predictions; points off
    the line show over- or under-prediction at a glance. The chart is also
    saved to docs/images/pred_vs_actual_teja.png for the README.
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from pathlib import Path
from typing import Final, cast

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "hours_scores_case"

# STEP 1. Pick the target variable we want to predict.

TARGET_COL: Final[str] = "score"

# STEP 2. Define the column names (features) that may help predict the target variable.

FEATURE_COLS: Final[list[str]] = [
    "hours_studied",
    "practice_quizzes",
    "attendance_pct",
    "sleep_hours",
    "prior_score",
]

# STEP 3. Define the test size and random state for reproducibility.

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# Where the Phase 4 chart is saved for the README.
CHART_PATH: Final[Path] = Path("docs") / "images" / "pred_vs_actual_teja.png"

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the case dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    # Select only the columns we need.
    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]

    # Drop rows with any missing values.
    df_no_missing: pd.DataFrame = df_selected.dropna()

    # Assign a copy of the no-missing DataFrame to df_clean to avoid SettingWithCopyWarning.
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(df_clean: pd.DataFrame) -> LinearRegression:
    """Train a supervised regression model."""
    LOG.info("Training LinearRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae: float = mean_absolute_error(y_test, y_pred)
    r2: float = r2_score(y_test, y_pred)

    LOG.info(f"Mean absolute error: {mae:.2f}")
    LOG.info(f"R-squared: {r2:.2f}")

    return model


# === Section 7. Predict One New Case ===


def predict_example(model: LinearRegression) -> None:
    """Use the trained model to predict one new student score."""
    LOG.info("Predicting one new case")

    new_case = pd.DataFrame(
        [
            {
                "hours_studied": 6.5,
                "practice_quizzes": 4,
                "attendance_pct": 92,
                "sleep_hours": 7.0,
                "prior_score": 72,
            }
        ]
    )

    predicted_score: float = model.predict(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted score: {predicted_score:.1f}")


# === Section 8. Create Visualizations ===


def make_plots(df_clean: pd.DataFrame, model: LinearRegression) -> None:
    """Create charts for the supervised regression case."""
    LOG.info("Creating chart: hours studied vs score")

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter_plt: Axes = sns.scatterplot(
        data=df_clean,
        x="hours_studied",
        y=TARGET_COL,
        ax=ax,
    )

    scatter_plt.set_title("Hours Studied vs Score (CLOSE chart to continue)")
    scatter_plt.set_xlabel("Hours Studied")
    scatter_plt.set_ylabel("Score")

    LOG.info("Creating chart: model coefficients")

    fig, ax = plt.subplots(figsize=(9, 5))

    coefficient_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_,
        }
    ).sort_values("coefficient", ascending=False)

    bar_plt: Axes = sns.barplot(
        data=coefficient_df,
        x="coefficient",
        y="feature",
        ax=ax,
    )

    bar_plt.set_title("Model Coefficients (CLOSE chart to continue)")
    bar_plt.set_xlabel("Coefficient")
    bar_plt.set_ylabel("Feature")

    # MY CHANGE: plot predicted vs actual score on the held-out test rows.
    # Re-create the same train/test split used in train_model() - TEST_SIZE
    # and RANDOM_STATE are fixed, so this reproduces the identical test rows.
    LOG.info("Creating chart: predicted vs actual score")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # WHY .to_numpy(): y_test keeps the original, non-sequential row index
    # from the train/test split, while model.predict() returns a plain array.
    # Passing the raw pandas Series in as x= lets seaborn align x and y by
    # index instead of by row position, silently dropping every test row
    # whose index doesn't happen to also appear in y_pred's default index.
    y_test_values = cast(pd.Series, y_test).to_numpy()
    y_pred = model.predict(x_test)

    fig, ax = plt.subplots(figsize=(9, 5))

    pred_vs_actual_plt: Axes = sns.scatterplot(
        x=y_test_values,
        y=y_pred,
        ax=ax,
    )

    # A perfect model would place every point on this diagonal line.
    line_min = min(min(y_test_values), min(y_pred))
    line_max = max(max(y_test_values), max(y_pred))
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="gray")

    pred_vs_actual_plt.set_title("Predicted vs Actual Score (CLOSE chart to continue)")
    pred_vs_actual_plt.set_xlabel("Actual Score")
    pred_vs_actual_plt.set_ylabel("Predicted Score")

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH)
    LOG.info(f"Saved chart to: {CHART_PATH}")


# === Section 9. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the supervised ML workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Create charts.............")
    make_plots(df_clean, model)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
