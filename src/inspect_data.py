from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

for csv_path in DATA_DIR.rglob("driving_log.csv"):
    df = pd.read_csv(csv_path, header=None)

    print(f"\n{'=' * 50}")
    print("Track:", csv_path.parent.name)

    raw_path = str(df.iloc[0, 0]).strip()
    recorded_path = Path(raw_path)

    filename = recorded_path.name
    expected_path = csv_path.parent / "IMG" / filename

    print("Raw CSV path:", repr(raw_path))
    print("Recorded path exists:", recorded_path.is_file())
    print("Expected path:", expected_path.resolve())
    print("Expected path exists:", expected_path.is_file())

    for col, camera in enumerate(["Center", "Left", "Right"]):
        paths = df[col].astype(str).str.strip()
        valid = sum(Path(p).is_file() for p in paths)

        print(f"{camera} images valid: {valid}/{len(df)}")

    actual_images = list((csv_path.parent / "IMG").glob("*.jpg"))
    print("Actual JPG files:", len(actual_images))

    if actual_images:
        print("Example actual image:", actual_images[0].resolve())