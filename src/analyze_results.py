
import torch
import numpy as np

from src.split_data import split_datasets
from src.evaluate import evaluate_model
from src.model import SteeringCNN
from torch.utils.data import DataLoader
from pathlib import Path

def get_subset_labels(subset):
    dataset = subset.dataset  
    all_labels = []
    for child_dataset in dataset.datasets:
        all_labels.extend(child_dataset.labels)

    all_labels = np.asarray(all_labels, dtype=np.float64)
    indices = np.asarray(subset.indices, dtype=np.int64)

    return all_labels[indices]

def calculate_metrics(predictions, actual):
    errors = predictions - actual

    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)

    return mse, mae, rmse


if __name__ == "__main__":
    BATCH_SIZE = 32
    CHECKPOINT_PATH = Path("models") / "best_model.pth"

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train your model first."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    train_dataset, val_dataset = split_datasets()
    train_labels = get_subset_labels(train_dataset)
    mean_steering = float(np.mean(train_labels))

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Mean training steering angle: {mean_steering:.6f}")


    actual = get_subset_labels(val_dataset)
    baseline_predictions = np.full_like(
        actual, mean_steering, dtype=np.float64
    )

    baseline_mse, baseline_mae, baseline_rmse = (
        calculate_metrics(baseline_predictions, actual)
    )

    model = SteeringCNN().to(device)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    cnn_predictions, cnn_actual, _ = evaluate_model(
        model,
        val_loader,
        device
    )

    cnn_mse, cnn_mae, cnn_rmse = calculate_metrics(
        cnn_predictions, cnn_actual
    )

    print("\nBaseline vs CNN")
    print("=" * 60)
    print(f"{'Metric':<12}{'Mean Baseline':>20}{'CNN':>15}")
    print("-" * 60)
    print(f"{'MSE':<12}{baseline_mse:>20.6f}{cnn_mse:>15.6f}")
    print(f"{'MAE':<12}{baseline_mae:>20.6f}{cnn_mae:>15.6f}")
    print(f"{'RMSE':<12}{baseline_rmse:>20.6f}{cnn_rmse:>15.6f}")

    print("\nCNN improvement over baseline")
    print("-" * 40)

    for name, baseline, cnn in [
        ("MSE", baseline_mse, cnn_mse),
        ("MAE", baseline_mae, cnn_mae),
        ("RMSE", baseline_rmse, cnn_rmse),
    ]:
        if baseline > 0:
            improvement = (baseline - cnn) / baseline * 100
            print(f"{name}: {improvement:+.2f}%")

    if cnn_mse < baseline_mse:
        print("\nThe CNN outperforms the mean-steering baseline on MSE.")
    elif cnn_mse > baseline_mse:
        print("\nThe mean-steering baseline performs better on MSE.")
    else:
        print("\nThe CNN and baseline have equal MSE.")
