from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = DOCS_DIR / "images"

NEAR_ZERO_LIMIT = 0.05
SHARP_TURN_LIMIT = 0.30
TOP_ERROR_COUNT = 10

DATASET_NAMES = ("driving_log.csv",)
TRAINING_HISTORY_NAMES = ("training-history.csv", "training_history.csv", "history.csv")
PREDICTION_NAMES = ("predictions.csv", "prediction_results.csv", "test_predictions.csv")
SIMULATOR_LOG_NAMES = ("simulator-logs.csv", "simulator_log.csv", "telemetry.csv")


def find_files(names, excluded_dirs=("venv", ".venv", ".git", "__pycache__", "site-packages")):
    matches = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.name.lower() not in {n.lower() for n in names}:
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        matches.append(path)
    return sorted(set(matches))


def resolve_output_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[plots] Skipping unreadable CSV {path}: {exc}")
        return None


def find_column(df, candidates):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None



def load_dataset(csv_path):
    df = read_csv(csv_path)
    if df is None:
        return None
    steering_col = find_column(
        df,
        ["steering", "steering_angle", "angle"],
    )
    if steering_col is None:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 4:
            df.columns = (
                ["center", "left", "right", "steering",
                 "throttle", "brake", "speed"][:df.shape[1]]
                if df.shape[1] <= 7
                else [
                    f"column_{i}"
                    for i in range(df.shape[1])
                ]
            )
            steering_col = "steering"
        else:
            print(
                f"[plots] Skipping {csv_path}: "
                "could not identify steering data."
            )
            return None
    df["steering"] = pd.to_numeric(
        df[steering_col],
        errors="coerce",
    )
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=["steering"])
    if df.empty:
        print(f"[plots] No valid steering values in {csv_path}")
        return None
    return df


def classify_steering(steering):
    magnitude = steering.abs()
    return pd.Series(
        np.select(
            [magnitude <= NEAR_ZERO_LIMIT, magnitude >= SHARP_TURN_LIMIT],
            ["Near-zero", "Sharp turn"],
            default="Moderate turn",
        ),
        index=steering.index,
    )


def plot_steering_distribution(track_datasets):
    if not track_datasets:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in track_datasets.items():
        ax.hist(df["steering"], bins=50, alpha=0.5, label=name)
    ax.set(title="Steering-Angle Distribution Across Datasets",
           xlabel="Steering angle", ylabel="Number of samples")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "steering-distribution.png")


def compare_steering_categories(track_datasets):
    if not track_datasets:
        return
    rows = []
    order = ["Near-zero", "Moderate turn", "Sharp turn"]
    for track, df in track_datasets.items():
        counts = classify_steering(df["steering"]).value_counts()
        for category in order:
            rows.append({"Track": track, "Category": category,
                         "Samples": int(counts.get(category, 0))})
    results = pd.DataFrame(rows)
    results.to_csv(DOCS_DIR / "steering-category-counts.csv", index=False)
    pivot = results.pivot(index="Category", columns="Track",
                          values="Samples").reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set(title="Steering Samples by Category",
           xlabel="Steering category", ylabel="Number of samples")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Dataset")
    save_figure(fig, "steering-categories.png")


def load_training_history(path):
    df = read_csv(path)
    if df is None:
        return None
    epoch = find_column(df, ["epoch", "epochs"])
    train = find_column(df, ["train_loss", "training_loss", "loss"])
    val = find_column(df, ["val_loss", "validation_loss", "valid_loss"])
    if train is None or val is None:
        print(f"[plots] Skipping {path}: need train_loss and val_loss columns.")
        return None
    df = df.copy()
    df["train_loss"] = pd.to_numeric(df[train], errors="coerce")
    df["val_loss"] = pd.to_numeric(df[val], errors="coerce")
    df["epoch"] = (
        pd.to_numeric(df[epoch], errors="coerce")
        if epoch is not None else np.arange(1, len(df) + 1)
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["epoch", "train_loss", "val_loss"]
    )
    return df if not df.empty else None


def plot_training_history(path):
    history = load_training_history(path)
    if history is None:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["epoch"], history["train_loss"], marker="o",
            label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o",
            label="Validation loss")
    ax.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "training-loss.png")


def load_prediction_results(path):
    df = read_csv(path)
    if df is None:
        return None
    actual_col = find_column(df, [
        "actual_steering", "actual", "y_true", "target", "steering"
    ])
    pred_col = find_column(df, [
        "predicted_steering", "predicted", "y_pred", "prediction"
    ])
    if actual_col is None or pred_col is None:
        print(f"[plots] Skipping {path}: need actual and predicted steering columns.")
        return None
    df = df.copy()
    df["actual_steering"] = pd.to_numeric(df[actual_col], errors="coerce")
    df["predicted_steering"] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual_steering", "predicted_steering"]
    )
    if df.empty:
        return None
    df["error"] = df["predicted_steering"] - df["actual_steering"]
    df["absolute_error"] = df["error"].abs()
    return df


def plot_prediction_analysis(path):
    results = load_prediction_results(path)
    if results is None:
        return

    actual = results["actual_steering"]
    predicted = results["predicted_steering"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.5, s=16, label="Predictions")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "--", label="Ideal: y = x")
    ax.set(title="Actual vs Predicted Steering",
           xlabel="Actual steering", ylabel="Predicted steering")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "prediction-scatter.png")

    mae = results["absolute_error"].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(results["absolute_error"], bins=50, alpha=0.8)
    ax.axvline(mae, linestyle="--", label=f"MAE = {mae:.4f}")
    ax.set(title="Absolute Prediction Error Distribution",
           xlabel="Absolute steering error", ylabel="Number of samples")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "error-distribution.png")

    largest = results.nlargest(TOP_ERROR_COUNT, "absolute_error").copy()
    largest.insert(0, "sample_index", largest.index)
    largest.to_csv(DOCS_DIR / "largest-prediction-errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(largest["sample_index"].astype(str), largest["absolute_error"])
    ax.set(title=f"Largest {len(largest)} Prediction Errors",
           xlabel="Sample index", ylabel="Absolute error")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "largest-prediction-errors.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(predicted, results["error"], alpha=0.5, s=16)
    ax.axhline(0, linestyle="--")
    ax.set(title="Residual Plot", xlabel="Predicted steering",
           ylabel="Residual (prediction - actual)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "residual-plot.png")

    actual_values = actual.to_numpy()
    predicted_values = predicted.to_numpy()
    baseline = np.full_like(actual_values, actual_values.mean())
    labels = ["CNN", "Mean baseline"]
    maes = [np.mean(np.abs(predicted_values - actual_values)),
            np.mean(np.abs(baseline - actual_values))]
    rmses = [np.sqrt(np.mean((predicted_values - actual_values) ** 2)),
             np.sqrt(np.mean((baseline - actual_values) ** 2))]
    pd.DataFrame({"Model": labels, "MAE": maes, "RMSE": rmses}).to_csv(
        DOCS_DIR / "baseline-comparison.csv", index=False
    )
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, maes, width, label="MAE")
    ax.bar(x + width / 2, rmses, width, label="RMSE")
    ax.set_xticks(x, labels)
    ax.set(title="CNN vs Mean-Prediction Baseline", ylabel="Error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "baseline-comparison.png")

    categories = classify_steering(results["actual_steering"])
    order = ["Near-zero", "Moderate turn", "Sharp turn"]
    results = results.copy()
    results["steering_range"] = categories
    summary = results.groupby("steering_range")["absolute_error"].agg(
        count="count", mean="mean", median="median"
    ).reindex(order)
    summary.to_csv(DOCS_DIR / "error-by-steering-range.csv")
    values = [
        results.loc[results["steering_range"] == category, "absolute_error"].to_numpy()
        for category in order
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(values, labels=order)
    ax.set(title="Prediction Error by Steering Category",
           xlabel="Actual steering category", ylabel="Absolute error")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "error-by-steering-range.png")

    print(f"[plots] Prediction metrics from {path}:")
    print(f"        Samples: {len(results)}")
    print(f"        MAE: {mae:.6f}")
    print(f"        RMSE: {np.sqrt(np.mean(results['error'] ** 2)):.6f}")


def plot_simulator_log(path):
    df = read_csv(path)
    if df is None:
        return
    time_col = find_column(df, ["time", "timestamp", "elapsed_seconds"])
    speed_col = find_column(df, ["speed", "speed_mph"])
    steering_col = find_column(df, ["steering", "steering_angle"])
    throttle_col = find_column(df, ["throttle"])
    if time_col is None or speed_col is None or steering_col is None:
        print(f"[plots] Skipping {path}: need time, speed, steering columns.")
        return
    for col in [time_col, speed_col, steering_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[time_col, speed_col, steering_col]
    )
    if df.empty:
        print(f"[plots] Skipping {path}: no valid telemetry rows.")
        return

    for column, title, ylabel, filename in [
        (speed_col, "Simulator Speed Over Time", "Speed (mph)", "speed-over-time.png"),
        (steering_col, "Simulator Steering Over Time", "Steering angle", "steering-over-time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[time_col], df[column])
        ax.set(title=title, xlabel="Time", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        save_figure(fig, filename)

    if throttle_col is not None:
        df[throttle_col] = pd.to_numeric(df[throttle_col], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[time_col], df[throttle_col])
        ax.set(title="Simulator Throttle Over Time",
               xlabel="Time", ylabel="Throttle")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "throttle-over-time.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plots] Project root: {PROJECT_ROOT}")
    print(f"[plots] Output folder: {OUTPUT_DIR}")

    dataset_files = find_files(DATASET_NAMES)
    track_datasets = {}
    for i, path in enumerate(dataset_files, start=1):
        df = load_dataset(path)
        if df is not None:
            track_datasets[f"{path.parent.name} ({i})"] = df
    if track_datasets:
        plot_steering_distribution(track_datasets)
        compare_steering_categories(track_datasets)
    else:
        print("[plots] No driving_log.csv with a steering column found.")

    histories = find_files(TRAINING_HISTORY_NAMES)
    if histories:
        plot_training_history(histories[0])
    else:
        print("[plots] No training history CSV found; skipping loss plot.")

    predictions = find_files(PREDICTION_NAMES)
    if predictions:
        plot_prediction_analysis(predictions[0])
    else:
        print("[plots] No prediction-results CSV found; skipping prediction plots.")

    simulator_logs = find_files(SIMULATOR_LOG_NAMES)
    if simulator_logs:
        plot_simulator_log(simulator_logs[0])
    else:
        print("[plots] No simulator telemetry CSV found; skipping simulator plots.")

    print("[plots] Finished. Available plots are in docs/images/.")


if __name__ == "__main__":
    main()
