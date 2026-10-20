from pathlib import Path
import glob
from src.dataset import CustomDataset
from torch.utils.data import ConcatDataset,Subset
from sklearn.model_selection import train_test_split

def split_datasets():
    DATA_DIR = Path("data")
    TRACK1_DIR = DATA_DIR / "track_1"
    TRACK2_DIR = DATA_DIR / "track_2"

    csv_path_track_1 = glob.glob(str(TRACK1_DIR / "*.csv"))
    csv_path_track_2 = glob.glob(str(TRACK2_DIR / "*.csv"))
    if not csv_path_track_1 or not csv_path_track_2:
        raise FileNotFoundError("CSV files For both directories are not found!")

    dataset1 = CustomDataset(csv_path_track_1, TRACK1_DIR/"IMG")
    dataset2 = CustomDataset(csv_path_track_2, TRACK2_DIR/"IMG")
    dataset = ConcatDataset([
        dataset1,
        dataset2
    ])
    indices = list(range(len(dataset)))
    train_indices,val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_dataset = Subset(dataset,train_indices)
    val_dataset = Subset(dataset,val_indices)
    return train_dataset,val_dataset

if __name__ == "__main__":
    train_dataset,val_dataset = split_datasets()
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    assert set(train_dataset.indices).isdisjoint(
        set(val_dataset.indices)
    )
    assert len(train_dataset) + len(val_dataset) == (len(train_dataset.dataset))
    print("DATA Split Successfully!")
