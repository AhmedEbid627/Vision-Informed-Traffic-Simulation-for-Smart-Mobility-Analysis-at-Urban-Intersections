from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FPS = 10.0
DEFAULT_WINDOW_FRAMES = 100  # 10 seconds at 10 fps.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute traffic flow rate from line-crossing data."
    )
    parser.add_argument(
        "--crossings-csv",
        type=Path,
        required=True,
        help="Path to a line-crossings CSV (track_id,frame_before,frame_after,...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to same directory as input.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help="Time window size in frames for binning crossings.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="Total frames in the video (for complete time range). If 0, inferred from data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.crossings_csv.exists():
        raise FileNotFoundError(f"Crossings CSV not found: {args.crossings_csv}")

    output_dir = args.output_dir or args.crossings_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.crossings_csv.stem.replace("_line_crossings", "")

    # Load crossing events.
    crossings: list[dict[str, str]] = []
    with args.crossings_csv.open("r", encoding="utf-8", newline="") as f:
        crossings = list(csv.DictReader(f))

    if not crossings:
        print("No crossing events found. Nothing to compute.")
        return

    # Extract crossing frames (use frame_after as the crossing moment).
    crossing_frames = [int(row["frame_after"]) for row in crossings]
    max_frame = args.total_frames if args.total_frames > 0 else max(crossing_frames) + 1

    # Bin crossings into time windows.
    window = args.window_frames
    bins: dict[int, int] = {}
    for bin_start in range(0, max_frame, window):
        bins[bin_start] = 0
    for frame in crossing_frames:
        bin_start = (frame // window) * window
        bins.setdefault(bin_start, 0)
        bins[bin_start] += 1

    sorted_bins = sorted(bins.items())
    window_seconds = window / args.fps

    # Write flow rate CSV.
    flow_path = output_dir / f"{stem}_flow_rate.csv"
    with flow_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_start_frame", "window_end_frame",
            "window_start_sec", "window_end_sec",
            "crossings", "flow_vehicles_per_min",
        ])
        for bin_start, count in sorted_bins:
            bin_end = min(bin_start + window, max_frame)
            start_sec = round(bin_start / args.fps, 2)
            end_sec = round(bin_end / args.fps, 2)
            actual_window_sec = (bin_end - bin_start) / args.fps
            flow_per_min = round((count / actual_window_sec) * 60, 4) if actual_window_sec > 0 else 0
            writer.writerow([bin_start, bin_end, start_sec, end_sec, count, flow_per_min])

    # Write summary.
    total_crossings = sum(c for _, c in sorted_bins)
    total_time_sec = max_frame / args.fps
    overall_flow = (total_crossings / total_time_sec) * 60 if total_time_sec > 0 else 0
    non_zero_bins = [c for _, c in sorted_bins if c > 0]

    summary_path = output_dir / f"{stem}_flow_rate_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_crossings", total_crossings])
        writer.writerow(["total_time_seconds", round(total_time_sec, 2)])
        writer.writerow(["overall_flow_veh_per_min", round(overall_flow, 4)])
        writer.writerow(["window_size_frames", window])
        writer.writerow(["window_size_seconds", round(window_seconds, 2)])
        writer.writerow(["max_crossings_in_window", max(non_zero_bins) if non_zero_bins else 0])
        writer.writerow(["windows_with_crossings", len(non_zero_bins)])
        writer.writerow(["total_windows", len(sorted_bins)])

    # Flow rate time-series plot.
    bin_mids = [(bs + window / 2) / args.fps for bs, _ in sorted_bins]
    counts = [c for _, c in sorted_bins]
    flows = []
    for bs, c in sorted_bins:
        be = min(bs + window, max_frame)
        ws = (be - bs) / args.fps
        flows.append((c / ws) * 60 if ws > 0 else 0)

    plt.figure(figsize=(12, 5))
    plt.bar(bin_mids, flows, width=window_seconds * 0.9, color="firebrick", edgecolor="white", alpha=0.85)
    plt.axhline(overall_flow, color="orange", linestyle="--", linewidth=1.5, label=f"Overall: {overall_flow:.1f} veh/min")
    plt.title("Traffic Flow Rate Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Flow Rate (vehicles/min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f"{stem}_flow_rate.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # Cumulative crossings plot.
    cum_crossings = np.cumsum(counts)
    times = [bs / args.fps for bs, _ in sorted_bins]
    plt.figure(figsize=(10, 5))
    plt.plot(times, cum_crossings, color="steelblue", linewidth=2)
    plt.title("Cumulative Vehicle Crossings Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Vehicles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cum_path = output_dir / f"{stem}_cumulative_crossings.png"
    plt.savefig(cum_path, dpi=160)
    plt.close()

    print(f"Flow rate CSV: {flow_path}")
    print(f"Flow summary CSV: {summary_path}")
    print(f"Flow rate plot: {plot_path}")
    print(f"Cumulative plot: {cum_path}")
    print(f"Total crossings: {total_crossings}")
    print(f"Overall flow: {overall_flow:.2f} vehicles/min")


if __name__ == "__main__":
    main()
