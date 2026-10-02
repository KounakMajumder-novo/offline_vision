# Limitations and honest status

- Detector: the default is a bundled OpenCV Zoo YOLOX-s model trained on COCO, run locally with OpenCV DNN. Upstream COCO validation results are not evidence of performance on this app's public-area scenes. The model is not trained specifically for dense crowds, does not recognize gestures, and may miss or duplicate people under occlusion. HOG remains an optional baseline.
- Industrial mode: object observations/events framework only; this prototype has no reliable component detector.
- Public-area mode uses temporary tracking IDs and fixed-camera image polygons; it is sensitive to camera movement, occlusion, crowded-scene ID switches, and incorrect zone configuration. It is not a validated crowd-safety or emergency-detection system.
- Hand gestures are not recognized by the default YOLOX-s person detector. Events are unavailable unless an explicitly configured detector has matching gesture classes; even then they report model labels only, not intent or threat.
- Face, identity, facial-expression, emotion, and gaze analysis are intentionally excluded. Event rules concern observed movement/zone behavior and must not label a person as risky.
- Operator-review drafts are local dashboard text only. No notification, emergency call, or third-party message is sent automatically. Conveyor-belt monitoring is outside the selected public-area scope.
- Tracking: greedy IoU is the default. Optional BoxMOT BoT-SORT provides Kalman tracking and camera-motion compensation; neither mode has been validated on representative footage for this project. IDs restart between runs.
- Lead-ahead output is a supplementary Kalman estimate from measured in-process detector/tracker latency, is capped at one second, and is advisory only. It is not a trajectory, navigation, or control estimate.
- Localization: unavailable until configured/calibrated; metric position requires a fixed-camera ground-plane homography. Pixel speed is not metric speed or GPS.
- Network: unauthenticated HTTP for a trusted LAN; no TLS or internet exposure.
- Not implemented: drone SDK, GPS/UWB/IMU, thermal sensor, phone-camera protocol, C++ acceleration, and manufacturing-specific model/data.
- Optional Gemma assistant requires a user-supplied Gemini API key and a model ID available to that API project. It sends a minimized scene summary to Google's hosted API only after a user action and explicit consent; summaries are not guaranteed accurate. Frame upload is off by default and requires both an operator environment flag and per-request consent.

No 3.5 ms target, production accuracy, public safety rating, industrial safety rating, or field performance is claimed. The dashboard and AI assistant do not control a drone or machine.
