import torch
from torch.utils.data import Dataset,DataLoader,ConcatDataset
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

    TRACK1_DIR = DATA_DIR / "track_1"
    FOLDER_PATH_TRACK1 = DATA_DIR / "IMG"
    csv_paths_track1 = glob.glob(str(TRACK1_DIR / "*.csv"))

    TRACK2_DIR = DATA_DIR / "track_2"
    FOLDER_PATH_TRACK2 = DATA_DIR / "IMG"
    csv_paths_track2 = glob.glob(str(TRACK2_DIR / "*.csv"))

    dataset_track1 = CustomDataset(csv_paths_track1,FOLDER_PATH_TRACK1)
    dataset_track2 = CustomDataset(csv_paths_track2,FOLDER_PATH_TRACK2)

    dataset = ConcatDataset([
        dataset_track1,
        dataset_track2
    ])

    print(f"Total Samples: {len(dataset)}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    images,labels = next(iter(dataloader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)


    assert images.shape == (32, 3, 66, 200)
    assert labels.shape == (32,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.float32

    assert torch.isfinite(images).all()
    assert torch.isfinite(labels).all()
    assert images.min() >= 0.0
    assert images.max() <= 1.0

    print("All DataLoader tests passed!")

    
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