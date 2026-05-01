from __future__ import annotations

from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parents[1]

# Choose the image you want to draw on.
image_path = project_root / "runs" / "track" / "infrastructure_batch" / "analysis" / "infrastructure_1000_queue_zone_preview.png"

window_name = "Draw Queue Zone"


def main() -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    clone = image.copy()
    selection = {"start": None, "end": None, "drawing": False}

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal image
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["start"] = (x, y)
            selection["end"] = (x, y)
            selection["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and selection["drawing"]:
            selection["end"] = (x, y)
            image = clone.copy()
            cv2.rectangle(image, selection["start"], selection["end"], (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            selection["end"] = (x, y)
            selection["drawing"] = False
            image = clone.copy()
            cv2.rectangle(image, selection["start"], selection["end"], (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Drag a rectangle with the mouse.")
    print("Press 'r' to reset, 'q' to quit, or Enter/Space to confirm.")

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            selection["start"] = None
            selection["end"] = None
            selection["drawing"] = False
            print("Selection reset.")
        elif key in (13, 32):  # Enter or Space
            if selection["start"] and selection["end"]:
                x1 = min(selection["start"][0], selection["end"][0])
                y1 = min(selection["start"][1], selection["end"][1])
                x2 = max(selection["start"][0], selection["end"][0])
                y2 = max(selection["start"][1], selection["end"][1])
                print(f"zone_x1 = {x1}")
                print(f"zone_y1 = {y1}")
                print(f"zone_x2 = {x2}")
                print(f"zone_y2 = {y2}")
            else:
                print("No rectangle selected yet.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
