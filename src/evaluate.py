from pathlib import Path
import torch
from src.model import SteeringCNN
from src.split_data import split_datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model,val_loader,device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_images = []
    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).reshape(-1,1)
            predictions = model(images)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            if len(all_images) < 8:
                remaining = 8 - len(all_images)
                all_images.extend(
                    images[:remaining].cpu()
                )
    predictions = np.array(all_predictions)
    actual = np.array(all_labels)
    mse = np.mean((predictions-actual) ** 2)
    mae = np.mean(np.abs(predictions-actual))
    rmse = np.sqrt(mse)
    print("\nEvaluation Results")
    print("-" * 45)
    print(f"Validation samples: {len(actual)}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("\nPredicted vs Actual Steering Values")
    print("-" * 45)

    for i in range(min(10, len(actual))):
        print(
            f"Sample {i + 1:02d} | "
            f"Predicted: {predictions[i]: .4f} | "
            f"Actual: {actual[i]: .4f} | "
            f"Absolute Error: {abs(predictions[i] - actual[i]):.4f}"
        )

    return predictions, actual, all_images


def visualize_predictions(images,predictions,actual):
    num_of_images = min(len(images),len(predictions),8)
    if (num_of_images == 0): 
        print("No Images Found!")
        return

    fig,axes = plt.subplots(2,4,figsize = (16,7))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= num_of_images:
            continue

        image = images[i].permute(1,2,0).numpy()
        axes[i].imshow(image)
        axes[i].set_title(
            f"Actual: {actual[i]:.3f} \n"
            f"Predicted: {predictions[i]:.3f}"
        )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CHECKPOINT_DIR = Path("models")/"best_model.pth"
    BATCH_SIZE = 32
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_DIR}\n"
            "Run src.train first to train and save the model."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = SteeringCNN().to(device=device)
    checkpoint = torch.load(
        CHECKPOINT_DIR,
        map_location=device
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    print(f"Loaded CHeckpoint: {CHECKPOINT_DIR}")
    print(f"BEST Checkpoint Epoch: {checkpoint['epoch']}")
    print(f"BEST Checkpoint Epoch: {checkpoint['val_loss']:.6f}")

    train_dataset,val_dataset = split_datasets()
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    predictions,actual,images = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device
    )
    visualize_predictions(
        images = images,
        predictions= predictions,
        actual= actual
    )
   