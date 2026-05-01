# Traffic Video to SUMO Simulation

## Presentation Structure

Use the presentation as a story. The cleanest division is:

### 1. Title Slide
- Project title
- Your name
- Course / supervisor / date

Suggested title:
`Traffic Flow Estimation from Intersection Video Using Computer Vision and SUMO Simulation`

### 2. Problem Motivation
- Why intersection traffic analysis matters
- Why manual observation is slow and limited
- Why combining computer vision with simulation is useful

Key message:
We want to extract traffic behavior from real video and use it to build a realistic simulation for analysis and future control testing.

### 3. Project Objective
- Detect vehicles in traffic scenes
- Track vehicles across frames
- Extract traffic metrics such as counts, flow, and queue behavior
- Use those metrics to drive a SUMO simulation

Key message:
The project is not only object detection. It is a video-to-simulation pipeline.

### 4. Dataset
- Infrastructure and Drone datasets
- Why Infrastructure was the main focus for simulation
- Original labels were multi-class (`car`, `bus`, `lorry`, `motorbike`)
- Converted to single class: `vehicle`

Key message:
The dataset choice was aligned with the smart-intersection use case.

### 5. Data Preparation
- COCO to YOLO conversion
- Visualization of labels to verify conversion
- Merging datasets
- Creating single-class datasets
- Validation split experiments

Key message:
A lot of work was done before training to make the dataset clean and usable.

### 6. Detection Model Development
- Trained YOLO on a combined single-class `vehicle` dataset
- Compared multi-class vs single-class training
- Single-class performed better for the project goal
- Tested larger validation split and extra dataset experiments

Key message:
A `vehicle` detector was more useful than fine-grained class recognition for traffic metrics.

### 7. Model Results
- Show training curves
- Show example detection images
- Explain that the model is good enough as a baseline, not perfect
- Mention that class imbalance motivated the single-class approach

Key message:
The model successfully detects vehicles and supports downstream tracking and analytics.

### 8. Tracking Pipeline
- Video built from image sequences
- Detection applied frame by frame
- ByteTrack used for persistent IDs
- Tracked video and CSV outputs generated

Key message:
Tracking is what turns detections into trajectories and traffic observations.

### 9. Traffic Metrics Extraction
- Vehicle counts per frame
- Line crossings
- Queue-zone occupancy
- Speed estimation
- Flow-rate estimation
- Travel-time estimation attempt

Key message:
The project moved from visual AI outputs to transportation-style metrics.

### 10. Batch Analysis Results
- Infrastructure sequences processed in batch
- Which sequences had the highest density / crossings / queue signal
- Why sequence `1000` became the main reference case

Key message:
Some clips were more useful for flow, some for congestion, and this helped identify the best case for simulation calibration.

### 11. SUMO Integration
- Generated SUMO network from OSM
- Built initial demand model from observed flow
- Ran SUMO and compared flow against observed values
- Improved route generation from random trips to curated valid approach-to-approach trips
- Refined routing to use the signalized intersection core so straight and turning movements are supported
- Added traffic-light timing scenarios and compared them against the observed flow case

Key message:
The simulation stage is already connected to the CV stage through observed traffic flow.

### 12. Current Simulation Results
- Simulated flow vs observed flow
- Show that flow is close
- Explain that route realism improved after replacing generic trips with valid intersection movements
- Show scenario comparison and identify the best-performing signal strategy

Key message:
The SUMO baseline is working, and the project can already test traffic-light strategies using CV-derived demand.

### 13. Current Limitations
- Travel-time validation is still weak
- Queue measurement is sensitive to region definition
- Simulation is calibrated mainly by flow so far
- The SUMO network still represents the signalized core better than the longer upstream/downstream corridors

Key message:
The pipeline is real and working, but not yet a fully calibrated digital twin.

### 14. Next Steps
- Extract directional movement counts from tracked trajectories
- Calibrate turning proportions using observed movements instead of assumed weights
- Improve network realism outside the immediate intersection core if needed
- Improve travel-time measurement
- Compare simulation and observation more deeply
- Test additional traffic-control scenarios if time allows

### 15. Conclusion
- Summarize what was achieved
- Emphasize the full pipeline from video to simulation
- Explain why this is a strong foundation for future smart-mobility analysis


## Report-Style Project Summary

## 1. Project Goal

The goal of the project is to estimate traffic behavior from real intersection video using computer vision, then use the extracted information to build and evaluate a SUMO traffic simulation.

This means the project has two connected parts:

1. Computer vision:
   detect and track vehicles in traffic footage.
2. Transportation simulation:
   convert observed traffic behavior into simulation inputs and evaluate whether the simulation reproduces realistic traffic conditions.

The important idea is that the project is not limited to image detection alone. The final objective is a complete video-to-simulation workflow.


## 2. Dataset Work

The initial dataset contained two major viewpoints:

- Infrastructure
- Drone

The infrastructure data was more suitable for the final use case because it resembles fixed traffic camera monitoring at an intersection.

The original annotations were multi-class. However, for this project the most important outputs were:

- vehicle counts
- tracking trajectories
- queue behavior
- flow behavior

Because of that, distinguishing `car`, `bus`, `lorry`, and `motorbike` was less important than simply detecting `vehicle`.

This led to an important design choice:

- Convert the dataset into a single-class `vehicle` detector

This change reduced the class imbalance problem and made the model more aligned with the actual transportation goal.

Major preparation steps completed:

- Converted COCO annotations to YOLO format
- Visualized YOLO labels to verify correctness
- Merged multiple datasets
- Built single-class datasets
- Created multiple train/validation configurations


## 3. Detection Model Development

Several training directions were explored:

- Multi-class detection
- Single-class `vehicle` detection
- Additional-dataset merging
- Alternative validation split experiments

The best practical conclusion was:

- Single-class `vehicle` detection is the strongest choice for this project

Why:

- It matches the downstream tasks better
- It avoids the worst effects of class imbalance
- It improves robustness for counting and tracking

The trained model became a usable baseline for video processing, even though it is not perfect.


## 4. Tracking and Video Pipeline

After training the detector, the next major step was to apply it to sequential traffic imagery.

Implemented pipeline:

- Build MP4 videos from image sequences
- Run detection on video frames
- Apply ByteTrack for multi-object tracking
- Export tracked videos
- Export tracking CSV files

This step was critical because it transformed isolated detections into tracked trajectories. That is what allowed the project to move from AI outputs to transportation metrics.


## 5. Traffic Metrics Extraction

Once tracking CSVs were available, multiple analysis scripts were built to estimate traffic quantities from the tracked trajectories.

The metrics implemented include:

- vehicle count per frame
- unique tracks
- line crossings
- queue-zone occupancy
- speed estimation in pixels/second
- flow-rate estimation in vehicles/minute
- travel-time estimation between reference lines

These outputs represent the bridge between the computer-vision stage and the simulation stage.

Important observation:

- Queue and travel-time metrics are more sensitive to geometry choices than plain counts or flow.

This means:

- counts and flow were easier to stabilize
- queue and travel time still need careful calibration


## 6. Batch Processing of Infrastructure Sequences

The infrastructure sequences were processed in batch rather than only one short clip.

This was important because it allowed comparison across multiple sequences and helped identify which clips were most useful for later simulation calibration.

From the batch analysis:

- sequence `0` had the highest line crossings
- sequence `3100` had the highest average vehicles per frame
- sequence `1000` stood out as the most useful reference clip for queue and flow interpretation

Sequence `1000` became the main reference case because:

- it had a strong flow signal
- it had a strong queue signal
- it was suitable for comparing observation against simulation


## 7. SUMO Integration

The project then moved into simulation.

What was already created:

- SUMO network from OSM
- SUMO configuration
- vehicle type definitions
- route generation script
- comparison script between SUMO outputs and observed data

The first demand-generation approach used `randomTrips.py`, driven by observed flow.

This gave a working simulation, but the routes were too generic and often unrealistic for the actual intersection behavior.

To improve this, the route generation logic was upgraded:

- inspect the network
- identify which approach-to-approach movements are actually routable
- first generate curated trips only along valid movements
- use `duarouter` to produce valid SUMO routes

At first, the valid movements were limited mainly to one corridor because some longer outer network chains were only partially routable. After deeper inspection, the signalized core of the intersection was identified as the correct routing area for scenario analysis.

The route generation logic was then improved again:

- use the four incoming core approaches of the signalized intersection
- use the four outgoing core departures
- support straight and turning movements through the traffic-light node
- keep only movements that are truly routable in the current SUMO network

This improved route realism substantially and made scenario testing more meaningful.


## 8. Current SUMO Status

The simulation is now functioning as a baseline and as a scenario-testing platform.

Most important current comparison:

- observed flow: about `6.54 veh/min`
- simulated flow: about `6.40 veh/min`

This is a close match, with only a small difference.

That is a strong result because it shows that the CV-derived traffic intensity is already useful for simulation calibration.

The most important upgrade is that the simulation no longer relies only on one west-east corridor. The current route model now supports straight and turning movements through the four-way signalized core of the intersection.

This means the simulation can now be used for traffic-light scenario testing in a way that is much closer to the actual project objective.

However, the current SUMO network still has an important limitation:

- the signalized core is represented better than the longer upstream and downstream road segments
- flow is already calibrated, but turning proportions are still assumed rather than directly estimated from video
- travel-time validation is still weaker than flow validation

### Scenario Results

Three traffic-light timing scenarios were tested using the same observed demand reference from sequence `1000`:

1. Baseline timing
2. Favor west-east movement
3. Favor north-south movement

Results:

- Baseline:
  flow `6.47 veh/min`, travel time `25.81 s`, wait time `15.12 s`
- Favor west-east:
  flow `6.52 veh/min`, travel time `24.23 s`, wait time `13.74 s`
- Favor north-south:
  flow `6.52 veh/min`, travel time `27.17 s`, wait time `16.29 s`

Interpretation:

- The `favor_west_east` scenario is currently the best one.
- It stays very close to the observed flow.
- It also reduces average travel time and waiting time relative to baseline.

This is an important project milestone because it shows that the pipeline is already capable of supporting a meaningful traffic-improvement analysis.


## 9. What Has Been Achieved So Far

The project already has a meaningful end-to-end foundation:

- dataset prepared
- detector trained
- detections validated visually
- tracking working
- traffic metrics extracted
- batch infrastructure analysis completed
- SUMO network created
- CV-derived flow used to generate SUMO demand
- simulation compared against observed flow
- route generation upgraded from generic random trips to intersection-valid movements
- signal-timing scenarios evaluated and compared

This means the project is already beyond a standard object-detection project. It is now a working prototype of a traffic observation and simulation framework.


## 10. What Still Needs To Be Done

The most important remaining work is calibration and realism improvement.

### A. Improve movement calibration from video

Reason:

- The current simulation already supports straight and turning movements through the core
- But the directional movement proportions are still assumed
- A stronger final result would estimate turning behavior from tracked video trajectories

This is now the most important next improvement for simulation realism.

### B. Improve network realism beyond the core

The current network works well for the signalized intersection core, but a later refinement could improve:

- longer approach roads
- outer network continuity
- realism of upstream/downstream vehicle behavior

This would make the simulation feel more like the full real-world scene, not just the intersection center.

### C. Improve travel-time estimation

The current travel-time measurement is not yet strong enough for serious validation.

It should be improved later once a clearer movement definition or better reference geometry is chosen.

### D. Improve queue calibration

Queue-zone analysis is already implemented, but it is sensitive to region definition and still needs careful interpretation.

### E. Compare observation vs simulation more deeply

Right now flow comparison is already useful.

The next stage should include:

- queue comparison
- travel-time comparison
- movement-specific comparison
- maybe lane- or direction-specific calibration


## 11. Suggested Final Message for the Presentation

The project successfully established a complete pipeline from traffic video to traffic simulation.

The most important achievement is not only that vehicles can be detected. The real contribution is that:

- traffic is observed from video,
- structured metrics are extracted,
- those metrics are used to drive a SUMO simulation,
- the simulation already shows a close match to observed flow,
- and alternative signal timings can now be tested and compared.

The remaining work is mainly about improving calibration, especially movement proportions, queue validation, and travel-time validation.

That means the project already has a strong core, and the next phase is refinement rather than starting from scratch.


## 12. Short Version You Can Say Verbally

This project built a pipeline that starts from traffic video, detects and tracks vehicles, extracts traffic metrics such as flow and queue behavior, and then uses those observations to build and evaluate a SUMO simulation. The current system already produces a simulation whose flow is close to the observed traffic flow, and it can already compare traffic-light timing scenarios. The main remaining challenge is improving calibration of movement patterns, queue behavior, and travel-time validation.
