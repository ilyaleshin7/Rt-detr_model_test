import time
from ultralytics import RTDETR

def main():

    model = RTDETR("runs/train/rtdetr_s_aug_3050ti5/weights/best.pt")

# ---------------------------
# 1. Анализ видео + измерение FPS
# ---------------------------
    video_path = "6366_City_Transportation_1920x1080.mp4"

    print(f"[INFO] Запускаю инференс на видео: {video_path}")

# Засекаем время
    start_time = time.time()
    frame_count = 0

    video_results = model.predict(
        source=video_path,
        save=True,        # сохранить результат в папку
        conf=0.3,         # порог уверенности
        stream=True,      # потоковый режим для видео
        project="video_results",
        name="detection"
        )

    for result in video_results:
        frame_count += 1  # считаем количество обработанных кадров

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"[RESULT] Видео обработано ✅")
    print(f"[RESULT] Всего кадров: {frame_count}")
    print(f"[RESULT] Время обработки: {elapsed_time:.2f} сек")
    print(f"[RESULT] Средний FPS: {fps:.2f}")

# ---------------------------
# 2. Метрики модели (mAP, precision, recall, F1)
# ---------------------------
    print("\n[INFO] Запускаю валидацию модели на встроенном датасете COCO  (для метрик)...")

    metrics = model.val(data="dataset/data.yaml")  # по умолчанию использует COCO val2017, если локально доступен
# Если хочешь валидировать на своём датасете → model.val(data="data.yaml")

    print("\n[RESULT] Метрики модели:")
    print(f"mAP50: {metrics.box.map50:.4f}")       # mAP@0.5
    print(f"mAP50-95: {metrics.box.map:.4f}")      # mAP@[.5:.95]
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print(f"F1-score: {metrics.box.f1:.4f}")

    print("\n[INFO] Тестирование завершено!")

if __name__ == "__main__":
    main()