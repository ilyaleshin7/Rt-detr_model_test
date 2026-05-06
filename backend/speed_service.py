from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import SPEED_LIMIT_CLASSES, ULTRALYTICS_CONFIG_DIR
from .schemas import Detection, PredictionResponse


@dataclass
class ActiveSpeedState:
    label: str | None = None
    speed_limit: int | None = None
    confidence: float | None = None
    message: str = ""

    @property
    def is_known(self) -> bool:
        return self.speed_limit is not None


class SpeedLimitService:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Any | None = None
        self.model_names: dict[int, str] = {}
        self.allowed_class_ids: set[int] = set()
        self.state = ActiveSpeedState()
        self._lock = Lock()

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.model_names = self._normalize_model_names(self.model.names)
        self.allowed_class_ids = {
            class_id
            for class_id, label in self.model_names.items()
            if label in SPEED_LIMIT_CLASSES
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def allowed_labels(self) -> list[str]:
        actual_labels = {self.model_names[class_id] for class_id in self.allowed_class_ids}
        return [label for label in SPEED_LIMIT_CLASSES if label in actual_labels]

    def predict(
        self,
        image_bytes: bytes,
        include_preview_detections: bool = False,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        image = self._read_image(image_bytes)
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        raw_detection_count = 0
        speed_detections: list[Detection] = []
        preview_detections: list[dict[str, Any]] = []

        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    raw_detection_count += 1
                    if include_preview_detections:
                        preview_detections.append(self._box_to_preview_detection(box))

                    detection = self._box_to_speed_detection(box)
                    if detection is not None:
                        speed_detections.append(detection)

        main_detection = self._select_main_detection(speed_detections)

        with self._lock:
            if main_detection is not None:
                response = self._apply_detected_speed(raw_detection_count, speed_detections, main_detection)
            else:
                response = self._apply_no_speed_detection(raw_detection_count)

        if include_preview_detections:
            response["preview_detections"] = preview_detections

        return response

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.is_loaded else "not_ready",
            "model_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "state": "known" if self.state.is_known else "unknown",
            "speed_limit": self.state.speed_limit,
            "allowed_speed_classes": self.allowed_labels,
        }

    def _apply_detected_speed(
        self,
        raw_detection_count: int,
        speed_detections: list[Detection],
        main_detection: Detection,
    ) -> PredictionResponse:
        label = main_detection["label"]
        speed_limit = SPEED_LIMIT_CLASSES[label]
        message = self._message_for_speed(speed_limit)
        display_should_update = self.state.speed_limit != speed_limit

        self.state = ActiveSpeedState(
            label=label,
            speed_limit=speed_limit,
            confidence=main_detection["confidence"],
            message=message,
        )

        return {
            "detected": raw_detection_count > 0,
            "speed_sign_detected": True,
            "detections": speed_detections,
            "main_sign": label,
            "speed_limit": speed_limit,
            "confidence": main_detection["confidence"],
            "message": message,
            "state": "known",
            "display_should_update": display_should_update,
        }

    def _apply_no_speed_detection(self, raw_detection_count: int) -> PredictionResponse:
        if self.state.is_known:
            return {
                "detected": raw_detection_count > 0,
                "speed_sign_detected": False,
                "detections": [],
                "main_sign": self.state.label,
                "speed_limit": self.state.speed_limit,
                "confidence": self.state.confidence,
                "message": self.state.message,
                "state": "known",
                "display_should_update": False,
            }

        return {
            "detected": raw_detection_count > 0,
            "speed_sign_detected": False,
            "detections": [],
            "main_sign": None,
            "speed_limit": None,
            "confidence": None,
            "message": "Сервис ещё не обнаружил знак ограничения скорости",
            "state": "unknown",
            "display_should_update": False,
        }

    def _box_to_speed_detection(self, box: Any) -> Detection | None:
        class_id = int(box.cls.item())
        if class_id not in self.allowed_class_ids:
            return None

        label = self.model_names[class_id]
        confidence = float(box.conf.item())
        bbox = [round(float(value), 2) for value in box.xyxy[0].tolist()]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": bbox,
        }

    def _box_to_preview_detection(self, box: Any) -> dict[str, Any]:
        class_id = int(box.cls.item())
        label = self.model_names.get(class_id, f"class_{class_id}")
        confidence = float(box.conf.item())
        bbox = [round(float(value), 2) for value in box.xyxy[0].tolist()]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": bbox,
            "is_speed_limit": class_id in self.allowed_class_ids,
        }

    @staticmethod
    def _normalize_model_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(label) for class_id, label in names.items()}

        return {class_id: str(label) for class_id, label in enumerate(names)}

    @staticmethod
    def _select_main_detection(detections: list[Detection]) -> Detection | None:
        if not detections:
            return None

        return max(detections, key=lambda detection: detection["confidence"])

    @staticmethod
    def _read_image(image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError("Загруженный файл не является корректным изображением") from exc

        return image.convert("RGB")

    @staticmethod
    def _message_for_speed(speed_limit: int) -> str:
        return f"Ограничение скорости {speed_limit} км/ч"
