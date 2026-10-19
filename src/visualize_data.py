from pathlib import Path
import pandas as pd
from PIL import Image
import glob, os
import matplotlib.pyplot as plt


def display_images(IMG_DIR,image_col,steering_col,df):
    fig,axes = plt.subplots(2,5,figsize = (18,7))
    axes = axes.flatten()
    df["image_name"] = df[image_col].astype(str).apply(
        lambda x: os.path.basename(x.strip())
    )

    image_paths = sorted(glob.glob(os.path.join(str(IMG_DIR), "*.jpg")))[:10]
    for i,ax in enumerate(axes):
        if i >= len(image_paths):
            ax.axis("off")
            continue

        image_path = Path(image_paths[i])
        image_name = image_path.name
        matching_rows = df[df["image_name"] == image_name]
        if not matching_rows.empty:
            steering = float(matching_rows.iloc[0][steering_col])
            with Image.open(image_path) as img:
                ax.imshow(img.copy())

            ax.set_title(
                f"Steering: {steering:.4f}",
                fontsize = 10
            )
            print(f"{image_name} -> Steering: {steering:.4f}")
        else:
            ax.set_title(f"Label Not Found\n{image_name}", fontsize = 8)
            print(f"{image_name} -> Steering Not Found!")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


## Looading some sample images
FOLDER_PATH_TRACK1 = r"data\track_1\IMG"
CSV_PATH_TRACK1 = r"data\track_1\driving_log.csv"
IMG_PATH_TRACK1 = sorted(glob.glob(os.path.join(FOLDER_PATH_TRACK1,"*.jpg")))[:10]
df1 = pd.read_csv(CSV_PATH_TRACK1,header=None)
df1.columns = [
    "center", "left", "right",
    "steering", "throttle", "brake", "speed"
]

FOLDER_PATH_TRACK2 = r"data\track_2\IMG"
CSV_PATH_TRACK2 = r"data\track_2\driving_log.csv"
IMG_PATH_TRACK2 = sorted(glob.glob(os.path.join(FOLDER_PATH_TRACK2,"*.jpg")))[:10]
df2 = pd.read_csv(CSV_PATH_TRACK2,header=None)
df2.columns = [
    "center", "left", "right",
    "steering", "throttle", "brake", "speed"
]

display_images(FOLDER_PATH_TRACK1,"center","steering",df1)
display_images(FOLDER_PATH_TRACK2,"center","steering",df2)

# df1 = pd.read_csv(CSV_PATH_TRACK1)

# print(df1.columns.tolist())
# print(df1.head())