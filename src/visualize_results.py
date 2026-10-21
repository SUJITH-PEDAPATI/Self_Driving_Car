
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

from src.model import SteeringCNN
from src.split_data import split_datasets


def evaluate_and_collect(model, val_loader, device, max_samples=10):
    model.eval()

    all_predictions = []
    all_actual = []
    all_images = []

    with torch.no_grad():
        for images, labels in val_loader:
            images_device = images.to(device)
            predictions = model(images_device)
            all_predictions.extend(
                predictions.cpu().numpy().reshape(-1)
            )
            all_actual.extend(
                labels.cpu().numpy().reshape(-1)
            )
            all_images.extend(
                images.cpu().unbind(dim=0)
            )

    predictions = np.asarray(all_predictions)
    actual = np.asarray(all_actual)

    absolute_errors = np.abs(predictions - actual)

    return predictions, actual, absolute_errors, all_images


def plot_actual_vs_predicted(actual, predictions):
    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        predictions,
        alpha=0.6,
        edgecolors="none"
    )

    min_value = min(actual.min(), predictions.min())
    max_value = max(actual.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--",
        label="Ideal prediction (y = x)"
    )

    plt.xlabel("Actual Steering Angle")
    plt.ylabel("Predicted Steering Angle")
    plt.title("Actual vs. Predicted Steering Angles")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_absolute_error_distribution(absolute_errors):
    plt.figure(figsize=(9, 5))

    plt.hist(
        absolute_errors,
        bins=30,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        np.mean(absolute_errors),
        linestyle="--",
        label=f"Mean error = {np.mean(absolute_errors):.4f}"
    )

    plt.xlabel("Absolute Prediction Error")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Absolute Steering Errors")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_largest_errors(
    images,
    actual,
    predictions,
    absolute_errors,
    top_n=10
):
    top_indices = np.argsort(absolute_errors)[::-1][:top_n]

    print("\nTop 10 Samples with Largest Errors")
    print("-" * 65)

    for rank, idx in enumerate(top_indices, start=1):
        print(
            f"{rank:02d}. Sample index: {idx:4d} | "
            f"Actual: {actual[idx]: .4f} | "
            f"Predicted: {predictions[idx]: .4f} | "
            f"Absolute error: {absolute_errors[idx]:.4f}"
        )

    number_of_images = len(top_indices)
    columns = 2
    rows = (number_of_images + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 4 * rows),
        squeeze=False
    )

    axes = axes.flatten()

    for plot_idx, sample_idx in enumerate(top_indices):
        ax = axes[plot_idx]

        # Convert image tensor (C, H, W) to (H, W, C)
        image = images[sample_idx].permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.set_title(
            f"Sample {sample_idx}\n"
            f"Actual: {actual[sample_idx]:.3f} | "
            f"Predicted: {predictions[sample_idx]:.3f}\n"
            f"Absolute error: {absolute_errors[sample_idx]:.3f}"
        )
        ax.axis("off")
    for i in range(number_of_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Validation Samples with Largest Steering Errors")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BATCH_SIZE = 32
    CHECKPOINT_PATH = Path("models") / "best_model.pth"

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train your model first."
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Selected device: {device}")

    model = SteeringCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    _, val_dataset = split_datasets()

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    predictions, actual, absolute_errors, images = (
        evaluate_and_collect(
            model,
            val_loader,
            device
        )
    )

    print(f"Validation samples: {len(actual)}")
    print(f"Mean absolute error: {absolute_errors.mean():.6f}")
    print(f"Maximum absolute error: {absolute_errors.max():.6f}")

    plot_actual_vs_predicted(actual, predictions)
    plot_absolute_error_distribution(absolute_errors)

    show_largest_errors(
        images,
        actual,
        predictions,
        absolute_errors,
        top_n=10
    )
