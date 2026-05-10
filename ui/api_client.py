from pathlib import Path
import time
from typing import Any

import cv2
import requests


class BackendClientError(RuntimeError):
    """Ошибка общения UI-клиента с backend-ом."""


class BackendClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict[str, Any]:
        response = self._request("get", "/health")
        return self._json_response(response)

    def predict_image_file(
        self,
        image_path: str | Path,
        include_preview_detections: bool = False,
    ) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise BackendClientError(f"Файл изображения не найден: {path}")

        with path.open("rb") as image_file:
            files = {"file": (path.name, image_file, "application/octet-stream")}
            response = self._request(
                "post",
                "/predict",
                files=files,
                params=self._preview_params(include_preview_detections),
            )

        return self._json_response(response)

    def predict_frame(
        self,
        frame: Any,
        include_preview_detections: bool = False,
        max_frame_width: int | None = None,
    ) -> dict[str, Any]:
        if frame is None:
            raise BackendClientError("Кадр для отправки пустой")

        original_height, original_width = frame.shape[:2]
        request_frame = self._resize_frame_for_request(frame, max_frame_width)
        request_height, request_width = request_frame.shape[:2]

        success, encoded_image = cv2.imencode(".jpg", request_frame)
        if not success:
            raise BackendClientError("Не удалось закодировать кадр в JPEG")

        files = {"file": ("frame.jpg", encoded_image.tobytes(), "image/jpeg")}
        started_at = time.perf_counter()
        response = self._request(
            "post",
            "/predict",
            files=files,
            params=self._preview_params(include_preview_detections),
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        payload = self._json_response(response)
        payload["_request_time_ms"] = round(elapsed_ms, 2)
        payload["_frame_scale_x"] = original_width / request_width if request_width else 1.0
        payload["_frame_scale_y"] = original_height / request_height if request_height else 1.0
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise BackendClientError("Backend недоступен. Проверьте, что FastAPI-сервис запущен.") from exc
        except requests.exceptions.Timeout as exc:
            raise BackendClientError("Backend не ответил за отведённое время.") from exc
        except requests.exceptions.HTTPError as exc:
            detail = self._extract_error_detail(exc.response)
            raise BackendClientError(f"Backend вернул ошибку: {detail}") from exc
        except requests.exceptions.RequestException as exc:
            raise BackendClientError(f"Ошибка запроса к backend-у: {exc}") from exc

        return response

    @staticmethod
    def _json_response(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendClientError("Backend вернул некорректный JSON.") from exc

        if not isinstance(payload, dict):
            raise BackendClientError("Backend вернул JSON неподдерживаемого формата.")

        return payload

    @staticmethod
    def _extract_error_detail(response: requests.Response | None) -> str:
        if response is None:
            return "нет ответа"

        try:
            payload = response.json()
        except ValueError:
            return response.text or f"HTTP {response.status_code}"

        if isinstance(payload, dict):
            return str(payload.get("detail", payload))

        return str(payload)

    @staticmethod
    def _preview_params(include_preview_detections: bool) -> dict[str, str] | None:
        if not include_preview_detections:
            return None

        return {"include_preview_detections": "true"}

    @staticmethod
    def _resize_frame_for_request(frame: Any, max_frame_width: int | None) -> Any:
        if max_frame_width is None or max_frame_width <= 0:
            return frame

        height, width = frame.shape[:2]
        if width <= max_frame_width:
            return frame

        scale = max_frame_width / width
        new_size = (max_frame_width, max(1, int(height * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
