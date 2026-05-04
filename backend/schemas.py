from typing import Literal, TypedDict


BBox = list[float]
StateName = Literal["unknown", "known"]


class Detection(TypedDict):
    label: str
    confidence: float
    bbox: BBox


class PredictionResponse(TypedDict):
    detected: bool
    speed_sign_detected: bool
    detections: list[Detection]
    main_sign: str | None
    speed_limit: int | None
    confidence: float | None
    message: str
    state: StateName
    display_should_update: bool
