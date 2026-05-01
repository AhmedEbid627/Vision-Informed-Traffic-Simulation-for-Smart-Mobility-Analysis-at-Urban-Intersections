# Vision-Informed Traffic Simulation for Smart Mobility Analysis at Urban Intersections

This repository contains a video-to-simulation workflow for intersection traffic analysis:

1. detect vehicles from traffic imagery,
2. track them over time,
3. extract traffic metrics such as counts, flow, queue occupancy, and relative motion,
4. use those observations to drive and evaluate a SUMO simulation.

The public repo is intentionally lightweight. Large datasets, training runs, generated videos, and presentation files are not included.


## What Is Included

- core Python scripts for dataset preparation, detection, tracking, metric extraction, and SUMO
- the SUMO network/configuration used in the project
- one small demo video under `examples/`
- one trained demo model under `models/`


## Quick Demo

The easiest way to try the project without the full dataset is to use the bundled demo assets:

- model: `models/traffic_vehicle_best.pt`
- sample video: `examples/sample_intersection.mp4`
- sample frame: `examples/sample_frame.jpg`

### Run detection on the sample video

```powershell
python .\scripts\run_video_detection.py
```

### Run tracking and export a CSV

```powershell
python .\scripts\run_video_tracking_to_csv.py
```

By default, both scripts use the included model and sample video. You can also override them:

```powershell
python .\scripts\run_video_tracking_to_csv.py `
  --model .\models\traffic_vehicle_best.pt `
  --video .\examples\sample_intersection.mp4
```


## Repository Structure

### Demo assets

- `examples/`
- `models/`

### Core scripts

- `scripts/convert_coco_to_yolo.py`
- `scripts/merge_yolo_datasets.py`
- `scripts/make_single_class_yolo.py`
- `scripts/train_yolo.py`
- `scripts/run_video_detection.py`
- `scripts/run_video_tracking.py`
- `scripts/run_video_tracking_to_csv.py`
- `scripts/analyze_tracking_csv.py`
- `scripts/count_line_crossings.py`
- `scripts/analyze_queue_zone.py`
- `scripts/compute_flow_rate.py`
- `scripts/estimate_speeds.py`
- `scripts/estimate_travel_time.py`
- `scripts/process_infrastructure_batch.py`
- `scripts/summarize_infrastructure_batch.py`
- `scripts/generate_sumo_routes.py`
- `scripts/compare_sumo_vs_observed.py`
- `scripts/run_sumo_signal_scenarios.py`
- `scripts/experiment_drone_support.py`

### SUMO setup

- `sumo_simulation/sumo_network.net.xml`
- `sumo_simulation/sumo_config.sumocfg`
- `sumo_simulation/sumo_routes.rou.xml`
- `sumo_simulation/vtypes.add.xml`
- `sumo_simulation/signal_scenarios/`


## Dataset Note

The full datasets are not uploaded here because they are too large for a normal source-code repository.

The project was developed using the MTID traffic dataset with infrastructure and drone viewpoints. If you want to reproduce training from scratch, you will need to download the dataset separately and recreate the YOLO-formatted data locally.

Because of that, this public repo is designed mainly for:

- understanding the workflow,
- running the included demo,
- inspecting the code,
- reproducing the SUMO setup,
- and extending the project with your own traffic data.


## Main Project Idea

The project is not only object detection.

The full pipeline is:

1. prepare traffic-intersection data,
2. train or fine-tune a single-class vehicle detector,
3. track vehicles across frames using ByteTrack,
4. compute traffic metrics from trajectories,
5. use observed flow to seed a SUMO simulation,
6. test traffic-signal timing strategies.


## Why Single-Class Vehicle Detection

The original traffic data contained multiple vehicle classes, but the transportation analysis mainly needed:

- counts,
- flow,
- queue behavior,
- and trajectories.

For that reason, the project collapsed classes into a single `vehicle` category to improve robustness and reduce class-imbalance issues.


## Current SUMO Scenario Result

Signal-timing scenarios were tested using the observed flow from the selected reference sequence.

| Scenario | Flow (veh/min) | Observed Flow (veh/min) | Avg Travel Time (s) | Avg Wait Time (s) |
|---|---:|---:|---:|---:|
| baseline | 6.47 | 6.54 | 25.81 | 15.12 |
| favor_west_east | 6.52 | 6.54 | 24.23 | 13.74 |
| favor_north_south | 6.52 | 6.54 | 27.17 | 16.29 |

Current takeaway:

- the simulated flow stays close to the observed flow,
- and the `favor_west_east` strategy currently performs best in this setup.


## Notes On The Public Repo

This repository intentionally excludes:

- raw datasets,
- YOLO training runs,
- generated tracking outputs,
- large presentation assets,
- and local cache folders.

That keeps the repository easier to browse and clone while still preserving the main code and simulation logic.

