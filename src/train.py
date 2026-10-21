import torch
from torch.utils.data import DataLoader
from src.model import SteeringCNN
from src.split_data import split_datasets
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU Not Found, Using CPU!")


def training_one_epoch(model,train_loader,criterion,optimizer,device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(
        train_loader,
        desc = "training",
        unit = "batch"
    )
    for images,labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).reshape(-1,1)
        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions,labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite training loss detected!"
            )
        loss.backward()
        optimizer.step()

        running_loss += (loss.item() * images.size(0))
        progress_bar.set_postfix(
            loss = f"{loss.item():.4f}"
        )
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate_one_epoch(model,val_loader,criterion,device):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(
        val_loader,
        desc = "validation",
        unit = "batch"
    )
    with torch.no_grad():
        for images,labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device).reshape(-1,1)
            predictions = model(images)
            loss = criterion(predictions,labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite validation loss detected!"
                )
            running_loss += (loss.item() * images.size(0))
            progress_bar.set_postfix(
                batch_loss = f"{loss.item():.4f}"
            )
    epoch_loss = running_loss/(len(val_loader.dataset))
    return epoch_loss

if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 16
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    CHECKPOINT_DIRECTORY = Path("models")
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    check_cuda_availability()
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")

    model = SteeringCNN()
    ## Moving the model onto the GPU
    model = model.to(device=device)

    ## Verfication of the Model Paramters
    # for name, parameter in model.named_parameters():
    #     print(f"{name}: {parameter.device}")
    #     assert parameter.device == device

    # print("Model successfully moved to", device)

    train_dataset,val_dataset = split_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        shuffle= True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr= LEARNING_RATE
    )
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = training_one_epoch(
            model = model,
            train_loader= train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device = device
        )
        val_loss =validate_one_epoch(\
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        print(
            f"Epoch: [{epoch+1}/{EPOCHS}] Summary |"
            f"Train Loss: {train_loss:.4f} |"
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                CHECKPOINT_DIRECTORY / "best_model.pth"
            )
            print("BEST model is Saved!")

    print("Training Completed Successfully!")
    print(f"Best Validation Loss: {best_val_loss}")
