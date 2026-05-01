from __future__ import annotations

import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "yolo_datasets" / "combined_vehicle" / "dataset.yaml"
runs_path = project_root / "runs" / "detect"
config_dir = project_root / ".ultralytics"

# Update these values directly in Visual Studio before running.
model_name = "yolo11s.pt"
epochs = 30
imgsz = 960
batch = 8
device = "0"
workers = 4
patience = 20
run_name = "traffic_vehicle_yolo11s"
cache = False
resume = False
seed = 42


def main() -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    runs_path.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    from ultralytics import YOLO

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(runs_path),
        name=run_name,
        cache=cache,
        resume=resume,
        seed=seed,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"Training finished. Outputs saved to: {save_dir}")
    else:
        print("Training finished.")


if __name__ == "__main__":
    main()
