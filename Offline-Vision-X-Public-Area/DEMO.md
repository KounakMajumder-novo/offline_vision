# Demo

For the clearly labeled synthetic public-area walkthrough (no camera required), install the supported dependencies and run:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 app.py --demo --host 127.0.0.1
```

The persistent **SIMULATED DATA · NOT LIVE** label, generated frame watermark, and API status distinguish fixture detections and sample zones from camera observations. It demonstrates the local UI/event path only; it is not evidence of detection or tracking performance and must not be used as a safety signal.

Warning rows include an incident-message draft for operator review only, not an automatic emergency call. **Copy draft** uses the browser clipboard when permitted; if clipboard access is blocked, the dashboard says so and leaves the draft selectable for manual copying. The app never transmits the draft.

For live camera input, run `py -3.13 app.py --host 127.0.0.1` and check `/api/status` for camera and detector availability. The bundled OpenCV Zoo YOLOX-s model supplies local COCO person detections to the session-local tracker; this was smoke-tested on an upstream multi-person sample, not field-validated. Configure actual camera polygons in `public_area.yaml` before using zone events. The dashboard reports camera or detector unavailability rather than substituting simulated detections.

To verify LAN viewing, connect a second device to the same private Wi-Fi and visit the laptop's LAN IP on port 8765.
