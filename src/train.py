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


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 16
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    CHECKPOINT_DIRECTORY = Path("models")
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    check_cuda_availability()
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
        batch_size=32,
        shuffle= True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr= LEARNING_RATE
    )
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        for images,labels in train_loader:
            images = images.to(device = device)
            labels = labels.to(device = device).reshape(-1,1)
            optimizer.zero_grad()

            prediction = model(images)
            loss = criterion(prediction,labels)
            
            ## Back Propagatioj
            loss.backward()
            ## Update the Model Weights
            optimizer.step()

            running_train_loss += (loss.item() * images.size(0))
        train_loss = running_train_loss/ (len(train_dataset))

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images,labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).reshape(-1,1)
                predictions = model(images)
                loss = criterion(predictions,labels)

                running_val_loss += (loss.item() * images.size(0))
        val_loss = running_val_loss / (len(val_dataset))

        print(
            f"Epoch: [{epoch+1}/{EPOCHS}]"
            f"Train Loss: {train_loss:.2f}"
            f"Validation Loss: {val_loss:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optmizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                CHECKPOINT_DIRECTORY / "best_model.pth"
            )
            print("BEST model is Saved!")

    print("Training Completed Successfully!")
    print(f"Best Validation Loss: {best_val_loss}")
