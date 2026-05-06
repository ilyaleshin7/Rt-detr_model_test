from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import MODEL_PATH, get_confidence_threshold
from .speed_service import SpeedLimitService


service = SpeedLimitService(
    model_path=MODEL_PATH,
    confidence_threshold=get_confidence_threshold(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load_model()
    yield


app = FastAPI(
    title="Backend MVP для ограничений скорости на YOLO26",
    description="Сервис распознавания знаков ограничения скорости по изображениям для дипломного MVP.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_preview_detections: bool = False,
) -> dict:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Загруженное изображение пустое")

    try:
        return service.predict(
            image_bytes,
            include_preview_detections=include_preview_detections,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
