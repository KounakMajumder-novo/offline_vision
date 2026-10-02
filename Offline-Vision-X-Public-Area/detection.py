"""Detector contract and checkpoint-backed custom detector."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Tuple


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # (x, y, width, height) in source-frame pixels
    bounding_box: Tuple[float, float, float, float]
    timestamp: float


class DetectorInterface(Protocol):
    def detect(self, image, timestamp: float = 0.0) -> List[Detection]: ...


class CustomGridDetector:
    """Runs the local, project-owned grid detector; never downloads weights."""
    def __init__(self, checkpoint: str, class_names: Sequence[str], confidence: float = 0.35,
                 nms_iou: float = 0.45, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is needed for detector inference; no remote fallback is used") from exc
        from model import GridDetector, decode
        self.torch = torch
        self.classes = list(class_names)
        self.confidence, self.nms_iou = confidence, nms_iou
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if state.get("class_names") != self.classes:
            raise ValueError("Checkpoint class list does not match configured class list")
        self.size, self.grid = int(state["image_size"]), int(state["grid_size"])
        self.model = GridDetector(len(self.classes), self.grid).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def detect(self, image, timestamp: float = 0.0) -> List[Detection]:
        import numpy as np
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype("uint8"))
        w, h = image.size
        resized = image.convert("RGB").resize((self.size, self.size))
        tensor = self.torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        with self.torch.no_grad():
            pred = self.model(tensor.to(self.device))[0].cpu()
        items = decode(pred, self.classes, w, h, self.confidence)
        items = _nms(items, self.nms_iou)
        return [Detection(i, self.classes[i], float(score), (x, y, bw, bh), timestamp)
                for i, score, x, y, bw, bh in items]


class UltralyticsYoloDetector:
    """Local-checkpoint YOLO detector; model loading never downloads weights."""

    def __init__(self, checkpoint: str, confidence: float = 0.25, image_size: int = 640,
                 device: str = "cpu", max_detections: int = 300):
        model_path = Path(checkpoint)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint does not exist: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is needed for YOLO inference; install requirements-tracking.txt"
            ) from exc
        if image_size < 1 or max_detections < 1:
            raise ValueError("image_size and max_detections must be positive")
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device
        self.max_detections = int(max_detections)
        names = self.model.names
        self.class_names: Dict[int, str] = (
            {int(key): str(value) for key, value in names.items()}
            if isinstance(names, dict)
            else {index: str(value) for index, value in enumerate(names)}
        )

    def detect(self, image, timestamp: float = 0.0) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        if not results:
            return []
        return self._convert_result(results[0], timestamp, self.class_names)

    @staticmethod
    def _convert_result(result, timestamp: float, class_names: Dict[int, str]) -> List[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        detections = []
        for coords, score, raw_class_id in zip(xyxy, scores, class_ids):
            x1, y1, x2, y2 = (float(value) for value in coords)
            class_id = int(raw_class_id)
            detections.append(Detection(
                class_id,
                class_names.get(class_id, f"class_{class_id}"),
                float(score),
                (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                timestamp,
            ))
        return detections


class OpenCVZooYoloXDetector:
    """Run OpenCV Zoo's pretrained COCO YOLOX-s detector locally."""

    class_names = {
        index: name for index, name in enumerate((
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ))
    }

    def __init__(self, checkpoint: str, confidence: float = 0.35, nms_iou: float = 0.5):
        model_path = Path(checkpoint)
        if not model_path.is_file():
            raise FileNotFoundError(f"OpenCV Zoo YOLOX model does not exist: {model_path}")
        try:
            from third_party.opencv_zoo_yolox.yolox import YoloX
        except ImportError as exc:
            raise RuntimeError("OpenCV Zoo YOLOX wrapper is missing") from exc
        self.model = YoloX(str(model_path), confThreshold=float(confidence),
                           nmsThreshold=float(nms_iou))

    def detect(self, image, timestamp: float = 0.0) -> List[Detection]:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(640 / height, 640 / width)
        resized = cv2.resize(
            rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded = np.full((640, 640, 3), 114.0, dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        rows = self.model.infer(padded)
        detections = []
        for x, y, box_width, box_height, confidence, raw_class_id in rows:
            class_id = int(raw_class_id)
            # COCO person is the only class relevant to public-area tracking.
            if class_id != 0:
                continue
            x1 = min(float(width), max(0.0, float(x / scale)))
            y1 = min(float(height), max(0.0, float(y / scale)))
            x2 = min(float(width), max(x1, float((x + box_width) / scale)))
            y2 = min(float(height), max(y1, float((y + box_height) / scale)))
            detections.append(Detection(
                class_id, self.class_names[class_id], float(confidence),
                (x1, y1, x2 - x1, y2 - y1),
                timestamp,
            ))
        return detections


def _nms(items, threshold):
    """Small dependency-free class-aware NMS. Items are (class, score, x,y,w,h)."""
    kept = []
    for cls in sorted({it[0] for it in items}):
        pending = sorted((it for it in items if it[0] == cls), key=lambda x: x[1], reverse=True)
        while pending:
            top = pending.pop(0); kept.append(top)
            def iou(a, b):
                ix1, iy1 = max(a[2], b[2]), max(a[3], b[3])
                ix2, iy2 = min(a[2]+a[4], b[2]+b[4]), min(a[3]+a[5], b[3]+b[5])
                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                return inter / max(1e-9, a[4]*a[5]+b[4]*b[5]-inter)
            pending = [x for x in pending if iou(top, x) <= threshold]
    return sorted(kept, key=lambda x: x[1], reverse=True)
