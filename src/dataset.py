import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from torchvision import transforms
import glob
import pandas as pd
from src.preprocess_image import convertToRGB,resizeImages,convertToTensors
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self,csv_paths,FOLDER_PATH):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([

        ])
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path,header=None)
            df.columns = [
                "center",
                "left",
                "right",
                "steering",
                "throttle",
                "brake",
                "speed"
            ]
            paths = df['center'].tolist()
            steering = df['steering'].tolist()
            for path,label in zip(paths,steering):
                image_path = Path(path)

                if not image_path.is_absolute():
                    image_path = Path(FOLDER_PATH) / image_path.name
                
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image = img.copy()

        image = convertToRGB([image])
        image = resizeImages(image)
        image = convertToTensors(image)
        image_tensor = image[0]

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image_tensor,label

if __name__ == "__main__":
    DATA_DIR = Path("data")
    FOLDER_PATH = DATA_DIR / "track_1"
    csv_paths = glob.glob(str(FOLDER_PATH / "*.csv"))
    dataset = CustomDataset(csv_paths,FOLDER_PATH)
    image,label = dataset[0]

    print(f"Image Shape: {image.shape}")
    print(f"label: {label}")

    assert image.shape == (3,66,200)
    assert image.dtype == torch.float32



    
    ## Debbugging code to check the CSV files and their contents
    # csv_paths = glob.glob(str(FOLDER_PATH / "*.csv"))
    # print("CSV files found:", csv_paths)

    # for csv_path in csv_paths:
    #     df = pd.read_csv(csv_path,header=None)
    #     df.columns = [
    #         "center",
    #         "left",
    #         "right",
    #         "steering",
    #         "throttle",
    #         "brake",
    #         "speed"
    #     ]
    #     print("CSV file:", csv_path)
    #     print("Column names:", df.columns.tolist())
    #     print(df.head())