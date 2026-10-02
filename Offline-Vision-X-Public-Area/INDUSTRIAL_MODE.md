# Industrial mode

Factory mode is a framework, not a validated inspection system. Configure zones and expected counts in `industrial.yaml` (or `config/industrial.yaml` in a packaged layout); supply suitable component detections from a trained, evaluated detector before relying on object-level alerts. Dwell, wrong-zone, unexpected-object, and missing-object rules are ordinary deterministic rules. Visual defect/anomaly detection requires representative labeled or normal-operation data and an evaluation plan. Alerts are advisory; the app does not control machinery.
