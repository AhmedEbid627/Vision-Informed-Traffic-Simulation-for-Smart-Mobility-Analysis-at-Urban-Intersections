from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ordered image sequence into an MP4 video."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing ordered image frames.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for the output video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    written = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable frame: {image_path}")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Wrote {written} frames to {output_path} at {args.fps} FPS")


if __name__ == "__main__":
    main()
