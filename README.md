# Traffic Video to SUMO Simulation

This project builds a pipeline that starts from traffic video, detects and tracks vehicles, extracts traffic metrics, and then uses those observations to drive and evaluate a SUMO traffic simulation.

The current project status is:

- vehicle detector trained for a single `vehicle` class
- video detection and tracking pipeline working
- traffic metrics extracted from infrastructure sequences
- SUMO network and demand generation working
- traffic-light timing scenarios tested against observed flow


## Project Goal

The goal is not only object detection.

The real objective is:

1. observe traffic behavior from video,
2. extract structured traffic metrics,
3. build a SUMO simulation from those observations,
4. test alternative traffic-control scenarios.


## Current Pipeline

### 1. Dataset Preparation

The project started from traffic image datasets with `Infrastructure` and `Drone` views.

Main preparation steps:

- convert COCO annotations to YOLO format
- visualize labels for sanity checking
- merge datasets
- convert to a single-class `vehicle` task

Why single class:

- the project mainly needs counts, tracking, flow, and queue behavior
- class-specific labels like `bus` vs `car` are less important than robust `vehicle` detection


### 2. Detection Model

The detection model was trained as a single-class `vehicle` detector.

This was chosen because it:

- reduces class imbalance problems
- improves robustness for downstream traffic analytics
- matches the project objective better than fine-grained vehicle classification


### 3. Video and Tracking

The project converts sequential image folders into videos, then applies:

- vehicle detection
- ByteTrack multi-object tracking
- tracked video export
- tracking CSV export

This turns frame-wise detections into usable vehicle trajectories.


### 4. Traffic Metrics

From the tracking CSV files, the project extracts:

- vehicle count per frame
- unique tracks
- line crossings
- queue-zone occupancy
- speed estimates
- flow-rate estimates

The infrastructure sequences were also processed in batch to compare traffic behavior across multiple clips.


### 5. SUMO Simulation

The SUMO stage includes:

- a network generated from OSM
- vehicle type definitions
- route generation based on observed flow
- simulation comparison against observed traffic
- scenario testing with alternative signal timings


## Important Folders

### Scripts

- `scripts/convert_coco_to_yolo.py`
- `scripts/merge_yolo_datasets.py`
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

### Outputs

- `runs/detect/`
- `runs/track/`
- `outputs/videos/`
- `sumo_simulation/`


## Most Important Analysis Outputs

### Infrastructure Batch Analysis

Main analysis outputs are under:

- `runs/track/infrastructure_batch/analysis/`
- `runs/track/infrastructure_batch/analysis/report/`

These contain:

- per-sequence tracking summaries
- flow summaries
- speed summaries
- queue summaries
- report tables and comparison plots


### SUMO Outputs

Main SUMO files are under:

- `sumo_simulation/sumo_network.net.xml`
- `sumo_simulation/sumo_routes.rou.xml`
- `sumo_simulation/sumo_tripinfo.xml`
- `sumo_simulation/sumo_comparison.csv`
- `sumo_simulation/signal_scenarios/`


## SUMO Route Model

The route generator was improved from a generic random-trip approach to a curated approach-to-approach model.

Important note:

- The project now uses the signalized intersection core for valid routing.
- This supports straight and turning movements through the intersection.
- The route generator filters to movements that are actually routable in the current network.

This makes the simulation more realistic than the earlier `randomTrips` baseline.


## Current SUMO Scenario Results

Signal-timing scenarios were tested using the observed flow from the `infrastructure_1000` sequence.

Results:

| Scenario | Flow (veh/min) | Observed Flow (veh/min) | Avg Travel Time (s) | Avg Wait Time (s) |
|---|---:|---:|---:|---:|
| baseline | 6.47 | 6.54 | 25.81 | 15.12 |
| favor_west_east | 6.52 | 6.54 | 24.23 | 13.74 |
| favor_north_south | 6.52 | 6.54 | 27.17 | 16.29 |

Interpretation:

- `favor_west_east` is currently the best scenario
- it keeps the simulated flow very close to the observed flow
- it also reduces average travel time and waiting time compared to baseline


## How to Run Important Parts

### Train Detector

```powershell
python .\scripts\train_yolo.py
```

### Run Tracking on a Video

```powershell
python .\scripts\run_video_tracking_to_csv.py
```

### Process Infrastructure Batch

```powershell
python .\scripts\process_infrastructure_batch.py
python .\scripts\summarize_infrastructure_batch.py
```

### Regenerate SUMO Routes from Observed Flow

```powershell
python .\scripts\generate_sumo_routes.py `
  --flow-summary .\runs\track\infrastructure_batch\analysis\infrastructure_1000_flow_rate_summary.csv `
  --net-file .\sumo_simulation\sumo_network.net.xml `
  --output-route .\sumo_simulation\sumo_routes.rou.xml `
  --sim-seconds 3600
```

### Run SUMO

```powershell
sumo -c .\sumo_simulation\sumo_config.sumocfg
```

### Open SUMO GUI

```powershell
sumo-gui -c .\sumo_simulation\sumo_config.sumocfg
```

### Compare SUMO Against Observed Flow

```powershell
python .\scripts\compare_sumo_vs_observed.py `
  --tripinfo .\sumo_simulation\sumo_tripinfo.xml `
  --observed .\runs\track\infrastructure_batch\analysis\infrastructure_1000_flow_rate_summary.csv `
  --output .\sumo_simulation\sumo_comparison.csv `
  --sim-seconds 3600
```

### Run Signal-Timing Scenarios

```powershell
python .\scripts\run_sumo_signal_scenarios.py
```


## Recommended Next Steps

The project is already a functioning prototype, but the most important next improvements are:

1. Extract directional movement counts from tracked trajectories
2. Improve observed queue validation against simulation
3. Improve travel-time measurement from video
4. Refine the SUMO network if longer upstream/downstream realism is needed
5. Use the scenario results in the final presentation and report


## Presentation Support

A longer presentation/report-style summary is available here:

- `docs/project_presentation_and_status_report.md`

This file contains:

- suggested presentation structure
- project summary
- completed work
- current results
- remaining tasks
