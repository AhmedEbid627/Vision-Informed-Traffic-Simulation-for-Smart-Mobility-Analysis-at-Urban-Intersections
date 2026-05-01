from __future__ import annotations

from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parents[1]

# Choose the image you want to draw on.
image_path = project_root / "runs" / "track" / "infrastructure_batch" / "analysis" / "infrastructure_1000_queue_zone_preview.png"

window_name = "Draw Queue Polygon"


def main() -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_image = cv2.imread(str(image_path))
    if base_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    points: list[tuple[int, int]] = []
    image = base_image.copy()

    def redraw() -> None:
        nonlocal image
        image = base_image.copy()
        for point in points:
            cv2.circle(image, point, 4, (0, 255, 0), -1)
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, (0, 255, 0), 2)
        if len(points) >= 3:
            cv2.line(image, points[-1], points[0], (0, 200, 255), 2)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Left-click to add polygon points.")
    print("Press 'u' to undo the last point, 'r' to reset, 'q' to quit, or Enter/Space to confirm.")

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if points:
                points.pop()
                redraw()
                print("Removed last point.")
        elif key == ord("r"):
            points.clear()
            redraw()
            print("Polygon reset.")
        elif key in (13, 32):  # Enter or Space
            if len(points) >= 3:
                print("polygon_points = [")
                for x, y in points:
                    print(f"    ({x}, {y}),")
                print("]")
            else:
                print("Select at least 3 points first.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
