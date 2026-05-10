import csv
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .api_client import BackendClient, BackendClientError
from .video_client import VideoWorker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "ui" / "assets"
UNKNOWN_ASSET = ASSETS_DIR / "unknown.png"


class TrafficSignWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.client = BackendClient()
        self.selected_image_path: Path | None = None
        self.selected_video_path: Path | None = None
        self.video_worker: VideoWorker | None = None
        self.current_asset_path: Path | None = None
        self.current_video_pixmap: QPixmap | None = None
        self.latest_metrics: dict[str, Any] | None = None
        self.latest_events: list[dict[str, Any]] = []
        self.has_synced_known_state = False

        self.setWindowTitle("MVP отображения ограничения скорости")
        self.resize(1180, 720)

        self.video_label = QLabel("Видео не выбрано")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(680, 420)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setObjectName("videoLabel")

        self.sign_label = QLabel()
        self.sign_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_label.setMinimumSize(260, 260)
        self.sign_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.message_label = QLabel("Ограничение неизвестно")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName("messageLabel")

        self.backend_status_label = QLabel("Backend: не проверен")
        self.backend_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.backend_status_label.setWordWrap(True)
        self.backend_status_label.setObjectName("statusLabel")

        self.metrics_status_label = QLabel("Метрики: нет данных")
        self.metrics_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_status_label.setWordWrap(True)
        self.metrics_status_label.setObjectName("statusLabel")

        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        self.file_path_input.setPlaceholderText("Файл не выбран")

        self.frame_stride_input = QSpinBox()
        self.frame_stride_input.setRange(1, 120)
        self.frame_stride_input.setValue(5)
        self.frame_stride_input.setSuffix(" кадр.")

        self.max_frame_width_input = QSpinBox()
        self.max_frame_width_input.setRange(0, 4096)
        self.max_frame_width_input.setSingleStep(160)
        self.max_frame_width_input.setValue(0)
        self.max_frame_width_input.setSuffix(" px")
        self.max_frame_width_input.setSpecialValueText("без resize")

        self.frame_stride_label = QLabel("Отправлять каждый:")
        self.frame_stride_label.setObjectName("settingsLabel")
        self.max_frame_width_label = QLabel("Макс. ширина кадра:")
        self.max_frame_width_label.setObjectName("settingsLabel")

        self.check_backend_button = QPushButton("Проверить backend")
        self.choose_image_button = QPushButton("Выбрать изображение")
        self.send_image_button = QPushButton("Отправить изображение")
        self.choose_video_button = QPushButton("Выбрать видео")
        self.start_video_button = QPushButton("Запустить видео")
        self.stop_video_button = QPushButton("Остановить видео")
        self.save_metrics_button = QPushButton("Сохранить метрики")

        self.send_image_button.setEnabled(False)
        self.start_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(False)
        self.save_metrics_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()
        self._apply_styles()
        QTimer.singleShot(0, lambda: self._set_unknown_display(force=True))

    def _build_layout(self) -> None:
        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(14)

        preview_frame = QFrame()
        preview_frame.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(QLabel("Предпросмотр видео"))
        preview_layout.addWidget(self.video_label, stretch=1)

        sign_frame = QFrame()
        sign_frame.setObjectName("panel")
        sign_frame.setMinimumWidth(330)
        sign_layout = QVBoxLayout(sign_frame)
        sign_title = QLabel("Текущее ограничение")
        sign_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sign_title.setObjectName("panelTitle")
        sign_layout.addWidget(sign_title)
        sign_layout.addWidget(self.sign_label, stretch=1)
        sign_layout.addWidget(self.message_label)
        sign_layout.addWidget(self.backend_status_label)
        sign_layout.addWidget(self.metrics_status_label)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(14)
        main_layout.addWidget(preview_frame, stretch=3)
        main_layout.addWidget(sign_frame, stretch=1)

        button_layout = QGridLayout()
        button_layout.setSpacing(8)
        button_layout.addWidget(self.check_backend_button, 0, 0)
        button_layout.addWidget(self.choose_image_button, 0, 1)
        button_layout.addWidget(self.send_image_button, 0, 2)
        button_layout.addWidget(self.choose_video_button, 1, 0)
        button_layout.addWidget(self.start_video_button, 1, 1)
        button_layout.addWidget(self.stop_video_button, 1, 2)
        button_layout.addWidget(self.save_metrics_button, 1, 3)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Файл:"))
        file_layout.addWidget(self.file_path_input)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(self.frame_stride_label)
        settings_layout.addWidget(self.frame_stride_input)
        settings_layout.addWidget(self.max_frame_width_label)
        settings_layout.addWidget(self.max_frame_width_input)
        settings_layout.addStretch(1)

        root_layout.addLayout(main_layout, stretch=1)
        root_layout.addLayout(file_layout)
        root_layout.addLayout(settings_layout)
        root_layout.addLayout(button_layout)

        self.setCentralWidget(central_widget)

    def _connect_signals(self) -> None:
        self.check_backend_button.clicked.connect(self.check_backend)
        self.choose_image_button.clicked.connect(self.choose_image)
        self.send_image_button.clicked.connect(self.send_image)
        self.choose_video_button.clicked.connect(self.choose_video)
        self.start_video_button.clicked.connect(self.start_video)
        self.stop_video_button.clicked.connect(self.stop_video)
        self.save_metrics_button.clicked.connect(self.save_metrics)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f6f8;
            }
            QFrame#panel {
                background: #ffffff;
                border: 1px solid #d8dee6;
                border-radius: 8px;
            }
            QLabel#videoLabel {
                background: #111820;
                color: #cbd5df;
                border-radius: 6px;
                font-size: 18px;
            }
            QLabel#panelTitle {
                color: #18202a;
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#messageLabel {
                color: #18202a;
                font-size: 24px;
                font-weight: 600;
            }
            QLabel#statusLabel {
                color: #52606d;
                font-size: 14px;
            }
            QLabel#settingsLabel {
                color: #18202a;
                font-size: 14px;
            }
            QPushButton {
                min-width: 170px;
                min-height: 36px;
                padding: 6px 12px;
                color: #18202a;
                background: #ffffff;
                border: 1px solid #b9c3cf;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #edf2f7;
            }
            QPushButton:disabled {
                color: #7a8794;
                background: #e3e8ee;
                border-color: #d2dae3;
            }
            QLineEdit {
                min-height: 30px;
                padding: 4px 8px;
                color: #18202a;
                background: #ffffff;
                border: 1px solid #b9c3cf;
                border-radius: 6px;
            }
            QSpinBox {
                min-height: 30px;
                padding: 4px 28px 4px 8px;
                color: #18202a;
                background: #ffffff;
                border: 1px solid #b9c3cf;
                border-radius: 6px;
            }
            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 24px;
                height: 17px;
                border-left: 1px solid #b9c3cf;
                border-bottom: 1px solid #d2dae3;
                border-top-right-radius: 6px;
                background: #f8fafc;
            }
            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 24px;
                height: 17px;
                border-left: 1px solid #b9c3cf;
                border-bottom-right-radius: 6px;
                background: #f8fafc;
            }
            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background: #edf2f7;
            }
            """
        )

    def check_backend(self) -> None:
        try:
            payload = self.client.health()
        except BackendClientError as exc:
            self.backend_status_label.setText(f"Backend: ошибка - {exc}")
            return

        status = payload.get("status", "unknown")
        model_loaded = payload.get("model_loaded")
        state = payload.get("state", "unknown")
        self.backend_status_label.setText(
            f"Backend: {status}; модель загружена: {model_loaded}; state: {state}"
        )

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать изображение",
            str(PROJECT_ROOT),
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp);;Все файлы (*)",
        )
        if not path:
            return

        self.selected_image_path = Path(path)
        self.file_path_input.setText(str(self.selected_image_path))
        self.send_image_button.setEnabled(True)

    def send_image(self) -> None:
        if self.selected_image_path is None:
            self.backend_status_label.setText("Изображение не выбрано")
            return

        try:
            payload = self.client.predict_image_file(self.selected_image_path)
        except BackendClientError as exc:
            self.backend_status_label.setText(f"Ошибка отправки изображения: {exc}")
            return

        self.apply_prediction(payload)
        self.backend_status_label.setText("Изображение обработано")

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать видео",
            str(PROJECT_ROOT),
            "Видео (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV);;Все файлы (*)",
        )
        if not path:
            return

        self.selected_video_path = Path(path)
        self.file_path_input.setText(str(self.selected_video_path))
        self.start_video_button.setEnabled(True)

    def start_video(self) -> None:
        if self.selected_video_path is None:
            self.backend_status_label.setText("Видео не выбрано")
            return

        if self.video_worker is not None and self.video_worker.isRunning():
            self.backend_status_label.setText("Видео уже обрабатывается")
            return

        self.latest_metrics = None
        self.latest_events = []
        self.save_metrics_button.setEnabled(False)
        self.metrics_status_label.setText("Метрики: обработка видео...")

        self.video_worker = VideoWorker(
            video_path=self.selected_video_path,
            client=self.client,
            send_every_n_frames=self.frame_stride_input.value(),
            max_frame_width=self.max_frame_width_input.value(),
        )
        self.video_worker.frame_ready.connect(self.update_video_frame)
        self.video_worker.prediction_received.connect(self.apply_prediction)
        self.video_worker.status_changed.connect(self.backend_status_label.setText)
        self.video_worker.error_occurred.connect(self._show_video_error)
        self.video_worker.metrics_ready.connect(self._store_metrics)
        self.video_worker.processing_finished.connect(self._video_finished)

        self.start_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)
        self.video_worker.start()

    def stop_video(self) -> None:
        if self.video_worker is not None and self.video_worker.isRunning():
            self.backend_status_label.setText("Остановка видео...")
            self.video_worker.stop()
            self.stop_video_button.setEnabled(False)
            if not self.video_worker.wait(3000):
                self.backend_status_label.setText("Ожидаю завершения текущего запроса к backend-у...")
                return

        self._video_finished()

    def apply_prediction(self, payload: dict[str, Any]) -> None:
        state = payload.get("state")

        if state == "unknown":
            self._set_unknown_display(force=True)
            return

        if state != "known":
            self.backend_status_label.setText(f"Неизвестное состояние backend-а: {state}")
            return

        should_update = bool(payload.get("display_should_update"))
        first_known_sync = not self.has_synced_known_state
        if not should_update and not first_known_sync:
            return

        main_sign = payload.get("main_sign")
        if not isinstance(main_sign, str) or not main_sign:
            self.backend_status_label.setText("Backend вернул state=known без main_sign")
            return

        asset_path = ASSETS_DIR / f"{main_sign}.png"
        if not asset_path.exists():
            self.backend_status_label.setText(f"PNG для знака не найден: {asset_path}")
            return

        message = payload.get("message") or self._message_from_speed(payload.get("speed_limit"))
        self._set_sign_asset(asset_path)
        self.message_label.setText(str(message))
        self.has_synced_known_state = True

    def update_video_frame(self, frame: Any) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel_count = rgb_frame.shape
        bytes_per_line = channel_count * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self.current_video_pixmap = QPixmap.fromImage(image)
        self._refresh_video_pixmap()

    def save_metrics(self) -> None:
        if self.latest_metrics is None:
            self.metrics_status_label.setText("Метрики: нет данных для сохранения")
            return

        directory = QFileDialog.getExistingDirectory(self, "Выбрать папку для сохранения", str(PROJECT_ROOT))
        if not directory:
            return

        output_dir = Path(directory)
        self._write_json(output_dir / "metrics.json", self.latest_metrics)
        self._write_metrics_csv(output_dir / "metrics.csv", self.latest_metrics)
        self._write_json(output_dir / "events.json", self.latest_events)
        self._write_events_csv(output_dir / "events.csv", self.latest_events)
        self.metrics_status_label.setText(f"Метрики сохранены: {output_dir}")

    def _store_metrics(self, metrics: dict[str, Any], events: list[dict[str, Any]]) -> None:
        metrics = dict(metrics)
        metrics["export_created_at"] = datetime.now().isoformat(timespec="seconds")
        metrics["video_path"] = str(self.selected_video_path) if self.selected_video_path else None
        self.latest_metrics = metrics
        self.latest_events = events
        self.save_metrics_button.setEnabled(True)
        self.metrics_status_label.setText(
            f"Метрики: кадров {metrics.get('total_frames')}, "
            f"анализировано {metrics.get('analyzed_frames')}, "
            f"FPS {metrics.get('estimated_processing_fps')}"
        )

    def _set_unknown_display(self, force: bool = False) -> None:
        if force or self.current_asset_path != UNKNOWN_ASSET:
            self._set_sign_asset(UNKNOWN_ASSET)

        self.message_label.setText("Ограничение неизвестно")
        self.has_synced_known_state = False

    def _set_sign_asset(self, asset_path: Path) -> None:
        pixmap = QPixmap(str(asset_path))
        if pixmap.isNull():
            self.backend_status_label.setText(f"Не удалось загрузить PNG: {asset_path}")
            return

        self.current_asset_path = asset_path
        self._refresh_sign_pixmap(pixmap)

    def _refresh_sign_pixmap(self, pixmap: QPixmap | None = None) -> None:
        if pixmap is None:
            if self.current_asset_path is None:
                return
            pixmap = QPixmap(str(self.current_asset_path))

        if pixmap.isNull():
            return

        target_size = self.sign_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.sign_label.setPixmap(scaled)

    def _refresh_video_pixmap(self) -> None:
        if self.current_video_pixmap is None or self.current_video_pixmap.isNull():
            return

        scaled = self.current_video_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh_sign_pixmap()
        self._refresh_video_pixmap()

    def _show_video_error(self, message: str) -> None:
        self.backend_status_label.setText(f"Ошибка видео: {message}")

    def _video_finished(self) -> None:
        self.stop_video_button.setEnabled(False)
        self.start_video_button.setEnabled(self.selected_video_path is not None)
        self.video_worker = None

    @staticmethod
    def _message_from_speed(speed_limit: Any) -> str:
        if speed_limit is None:
            return "Ограничение неизвестно"

        return f"Ограничение скорости {speed_limit} км/ч"

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                if isinstance(value, list):
                    value = ";".join(str(item) for item in value)
                writer.writerow([key, value])

    @staticmethod
    def _write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["frame", "event", "message", "data"])
            writer.writeheader()
            for event in events:
                row = dict(event)
                row["data"] = json.dumps(row.get("data", {}), ensure_ascii=False)
                writer.writerow(row)


def main() -> int:
    app = QApplication(sys.argv)
    window = TrafficSignWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
