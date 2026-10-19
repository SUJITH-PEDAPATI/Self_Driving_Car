## Loading the images from both the tracks
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import glob
import matplotlib.pyplot as plt

def load_images(FOLDER_PATH: str) -> list:
    images = []
    image_path = sorted(glob.glob(os.path.join(str(FOLDER_PATH), "*.jpg")))
    for image in image_path:
        try:
            with Image.open(image) as img:
                images.append(img.copy())
        except Exception as e:
            print(f"CouldNot Load Image! {e}")
    
    return images


def convertToRGB(images: list) -> list:
    outputModifiedImages = []
    for image in images:
        img = image.convert("RGB")
        cropped_image = img.crop((0,60,320,140))
        outputModifiedImages.append(cropped_image)
    
    return outputModifiedImages


def resizeImages(images: list) -> list:
    outputResizedImages = []
    for image in images:
        resized_image = image.resize((200,66))
        outputResizedImages.append(resized_image)
    
    return outputResizedImages


def convertToTensors(images: list) -> list:
    transform = transforms.ToTensor()
    tensors = []

    for image in images:
        tensor = transform(image)
        tensors.append(tensor)

    return tensors

def verifyTensor(tensor):
    print(f"Tensor dtype", tensor.dtype)
    print(f"Tensor shape", tensor.shape)
    print(f"Minimum Value", tensor.min().item())
    print(f"Maximum Value", tensor.max().item())

    assert tensor.shape == (3,66,200), "Unexpected Tensor Shape"
    assert tensor.is_floating_point(), "Tensor must be a floating point"
    assert tensor.min().item() >= 0.0, "Values are below 0!"
    assert tensor.max().item() <= 1.0, "Values above 1!"


def display_image(image):
    plt.imshow(image)
    plt.title("Processed Road Image")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    DATA_DIR = Path("data")
    FOLDER_PATH_TRACK1 = DATA_DIR / "track_1" / "IMG"
    FOLDER_PATH_TRACK2 = DATA_DIR / "track_2" / "IMG"

    track1_images = load_images(FOLDER_PATH_TRACK1)
    track2_images = load_images(FOLDER_PATH_TRACK2)

    track1_images = convertToRGB(track1_images)
    track2_images = convertToRGB(track2_images)

    track1_images = resizeImages(track1_images)
    track2_images = resizeImages(track2_images)

    track1_tensors = convertToTensors(track1_images)
    track2_tensors = convertToTensors(track2_images)

    if track1_tensors: verifyTensor(track1_tensors[0])
    if track2_tensors: verifyTensor(track2_tensors[0])
    if track1_images: display_image(track1_images[0])
    if track2_images: display_image(track2_images[0])