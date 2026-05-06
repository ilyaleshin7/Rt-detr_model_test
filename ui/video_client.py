from pathlib import Path
import time
from typing import Any

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .api_client import BackendClient, BackendClientError


class VideoWorker(QThread):
    frame_ready = pyqtSignal(object)
    prediction_received = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    metrics_ready = pyqtSignal(dict, list)
    processing_finished = pyqtSignal()

    def __init__(
        self,
        video_path: str | Path,
        client: BackendClient,
        send_every_n_frames: int = 5,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.client = client
        self.send_every_n_frames = max(1, send_every_n_frames)
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            message = f"Не удалось открыть видео: {self.video_path}"
            self.error_occurred.emit(message)
            self.metrics_ready.emit(self._empty_metrics(), [self._event(0, "error", message)])
            self.processing_finished.emit()
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        delay_ms = int(1000 / fps) if fps and fps > 0 else 30
        frame_index = 0
        total_frames = 0
        analyzed_frames = 0
        inference_times_ms: list[float] = []
        total_speed_detections = 0
        speed_sign_detected_frames = 0
        state_changes_count = 0
        unique_detected_speed_classes: set[str] = set()
        events: list[dict[str, Any]] = []
        last_preview_detections: list[dict[str, Any]] = []
        started_at = time.perf_counter()

        self.status_changed.emit("Видео запущено")

        try:
            while not self._stop_requested:
                success, frame = capture.read()
                if not success:
                    events.append(self._event(frame_index, "video_finished", "Видео завершено"))
                    self.status_changed.emit("Видео завершено")
                    break

                total_frames += 1

                if frame_index % self.send_every_n_frames == 0:
                    try:
                        prediction = self.client.predict_frame(
                            frame,
                            include_preview_detections=True,
                        )
                    except BackendClientError as exc:
                        message = str(exc)
                        events.append(self._event(frame_index, "error", message))
                        self.error_occurred.emit(message)
                        break

                    analyzed_frames += 1
                    inference_times_ms.append(float(prediction.get("_request_time_ms", 0.0)))
                    speed_detections = prediction.get("detections", [])
                    total_speed_detections += len(speed_detections)

                    for detection in speed_detections:
                        label = detection.get("label")
                        if isinstance(label, str):
                            unique_detected_speed_classes.add(label)
                            events.append(
                                self._event(
                                    frame_index,
                                    "speed_detected",
                                    f"Обнаружен {label}",
                                    {"label": label, "confidence": detection.get("confidence")},
                                )
                            )

                    if prediction.get("speed_sign_detected"):
                        speed_sign_detected_frames += 1

                    if prediction.get("state") == "known":
                        main_sign = prediction.get("main_sign")
                        if prediction.get("display_should_update"):
                            state_changes_count += 1
                            events.append(
                                self._event(
                                    frame_index,
                                    "active_sign_changed",
                                    f"Активный знак изменён на {main_sign}",
                                    {"main_sign": main_sign, "speed_limit": prediction.get("speed_limit")},
                                )
                            )
                        elif not prediction.get("speed_sign_detected") and main_sign:
                            events.append(
                                self._event(
                                    frame_index,
                                    "speed_not_found_keep_active",
                                    f"Speed-limit sign не найден, сохранён {main_sign}",
                                    {"main_sign": main_sign, "speed_limit": prediction.get("speed_limit")},
                                )
                            )
                    elif prediction.get("state") == "unknown":
                        events.append(self._event(frame_index, "state_unknown", "state unknown"))

                    last_preview_detections = self._preview_detections_from_prediction(prediction)
                    self.prediction_received.emit(prediction)

                preview_frame = self._draw_detections(frame, last_preview_detections)
                self.frame_ready.emit(preview_frame)

                frame_index += 1
                self.msleep(max(1, delay_ms))

            if self._stop_requested:
                events.append(self._event(frame_index, "video_stopped", "Видео остановлено пользователем"))
                self.status_changed.emit("Видео остановлено")
        finally:
            capture.release()
            elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
            metrics = {
                "total_frames": total_frames,
                "analyzed_frames": analyzed_frames,
                "send_every_n_frames": self.send_every_n_frames,
                "avg_inference_time_ms": round(sum(inference_times_ms) / len(inference_times_ms), 2)
                if inference_times_ms
                else 0.0,
                "estimated_processing_fps": round(total_frames / elapsed_seconds, 2),
                "total_speed_detections": total_speed_detections,
                "speed_sign_detected_frames": speed_sign_detected_frames,
                "state_changes_count": state_changes_count,
                "unique_detected_speed_classes": sorted(unique_detected_speed_classes),
            }
            self.metrics_ready.emit(metrics, events)
            self.processing_finished.emit()

    def _empty_metrics(self) -> dict[str, Any]:
        return {
            "total_frames": 0,
            "analyzed_frames": 0,
            "send_every_n_frames": self.send_every_n_frames,
            "avg_inference_time_ms": 0.0,
            "estimated_processing_fps": 0.0,
            "total_speed_detections": 0,
            "speed_sign_detected_frames": 0,
            "state_changes_count": 0,
            "unique_detected_speed_classes": [],
        }

    @staticmethod
    def _preview_detections_from_prediction(prediction: dict[str, Any]) -> list[dict[str, Any]]:
        detections = prediction.get("preview_detections")
        if isinstance(detections, list):
            return [detection for detection in detections if isinstance(detection, dict)]

        speed_only = prediction.get("detections")
        if isinstance(speed_only, list):
            return [detection for detection in speed_only if isinstance(detection, dict)]

        return []

    @staticmethod
    def _draw_detections(frame: Any, detections: list[dict[str, Any]]) -> Any:
        annotated = frame.copy()
        for detection in detections:
            bbox = detection.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(float(value)) for value in bbox]
            is_speed_limit = bool(detection.get("is_speed_limit", True))
            color = (40, 190, 80) if is_speed_limit else (30, 150, 240)
            label = str(detection.get("label", "class"))
            confidence = detection.get("confidence")
            text = f"{label} {float(confidence):.2f}" if isinstance(confidence, (int, float)) else label

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            text_y = max(0, y1 - text_h - 8)
            cv2.rectangle(annotated, (x1, text_y), (x1 + text_w + 8, text_y + text_h + 8), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 4, text_y + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def _event(
        frame: int,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "frame": frame,
            "event": event_type,
            "message": message,
            "data": data or {},
        }
