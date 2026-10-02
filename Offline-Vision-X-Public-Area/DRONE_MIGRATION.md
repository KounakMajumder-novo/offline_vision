# Drone / factory tracking integration

## Evaluated options

| Option | Fit | Limitation |
|---|---|---|
| [Ultralytics tracking mode](https://docs.ultralytics.com/modes/track/) with `tracker="botsort.yaml"` | Most direct YOLO path; BoT-SORT is supported alongside ByteTrack and can keep IDs across frames. | Detector and tracking lifecycle are coupled through `model.track`, which does not preserve this app's separate detector-to-tracker seam as directly. |
| [BoxMOT BoT-SORT](https://github.com/mikel-brostrom/boxmot/blob/master/docs/trackers/botsort.md) with an independent Ultralytics detector | Chosen. BoxMOT documents standalone `N x 6` AABB detections (`xyxy, confidence, class_id`) and `M x 8` track rows. Its BoT-SORT includes Kalman motion estimation and camera-motion compensation, useful for moving-camera footage; the existing app can keep its detector and API boundaries. | Optional Torch-based dependencies and more setup; current BoxMOT docs support Python 3.10–3.13. A frame is passed for camera-motion compensation. ReID embeddings are disabled in this integration to avoid implicit model-weight acquisition. |
| [BoxMOT ByteTrack](https://github.com/mikel-brostrom/boxmot/blob/master/docs/trackers/bytetrack.md) | Lighter association option for fixed industrial cameras and lower-compute systems. | Not the selected common tracker: the drone view may move, while the selected BoT-SORT path offers camera-motion compensation and optional appearance association. |

The decision is based on documented input/output contracts and architecture fit, not measured speed or UAV/factory accuracy. BoxMOT's benchmark results are not treated as evidence for this project's camera, classes, or hardware. The BoT-SORT algorithm reference is [the original paper](https://arxiv.org/abs/2206.14651).

## Local configuration

The default detector is the bundled OpenCV Zoo YOLOX-s COCO checkpoint with the existing greedy-IoU tracker; inference is local and needs no separate model download. COCO's upstream evaluation is not a measured score for this application. To select the optional BoT-SORT + custom YOLO integration:

1. Install optional dependencies with `python -m pip install -r requirements-tracking.txt` on Python 3.10–3.13.
2. Put a compatible, locally obtained YOLO detection checkpoint at `models/yolo.pt`.
3. Set `detector.yaml`'s `backend` to `ultralytics_yolo`; set `tracker.yaml`'s `backend` to `boxmot_botsort`.
4. Run the existing `python app.py` entry point.

The app checks that the configured checkpoint exists before constructing YOLO, and never asks Ultralytics to fetch a missing checkpoint. The adapter converts the model's `xyxy/conf/class` values to the BoxMOT AABB6 contract. BoxMOT's Kalman tracker output is separate from the app's supplementary `LatencyKalmanProjector`, which projects the image-space box center ahead by measured in-process latency, capped at one second.

## Limits and field validation

The `DRONE` and `FACTORY` HUD modes are visual monitoring labels, not flight or machine telemetry. The API explicitly marks projection as `advisory_only` and reports `control_enabled: false`. This repo does not connect to an autopilot, GPS/IMU, gimbal, factory PLC, or safety interlock; it must not issue control commands. Without calibrated image-to-plane mapping, position and lead-ahead values are pixels; a homography is not GPS and assumes a fixed camera and plane.

Stock YOLO weights do not imply drone/UAV or factory-part recognition. Supply and validate a model trained for the target class set. Before field use, test recorded sequences for detector precision/recall, identity continuity, occlusion, camera motion, gimbal movement, rolling shutter, altitude/scale changes, localization error, and end-to-end latency on the target hardware. The displayed latency is measured after frame capture through detection and tracking only. No 3.5 ms target, production accuracy, safety rating, or flight performance is claimed.
