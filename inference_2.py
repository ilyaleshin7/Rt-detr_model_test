import os
import time
from ultralytics import YOLO

def main():
    # ---------------------------
    # 0. Пути
    # ---------------------------
    model_path = "best_weights/rtdetr_best.pt"
    video_path = "video/видос проверка.MOV"
    data_yaml = "new_dataset_yolo26/data.yaml"   # поменяй, если data.yaml лежит в другом месте

    # Папка, куда сохранится видео с инференсом
    save_project = "video_results"
    save_name = "detection"

    # ---------------------------
    # Проверка файлов
    # ---------------------------
    if not os.path.exists(model_path):
        print(f"[ERROR] Не найден файл модели: {model_path}")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] Не найден видеофайл: {video_path}")
        return

    print(f"[INFO] Загружаю модель: {model_path}")
    model = YOLO(model_path)

    # ---------------------------
    # 1. Анализ видео + измерение FPS
    # ---------------------------
    print(f"[INFO] Запускаю инференс на видео: {video_path}")

    start_time = time.time()
    frame_count = 0

    video_results = model.predict(
        source=video_path,
        save=True,               # сохранить результат
        conf=0.3,                # порог уверенности
        stream=True,             # потоковый режим для видео
        project="video_results",    # папка сохранения
        name="detection",          # подпапка
        imgsz=640,
        verbose=False
    )

    for _ in video_results:
        frame_count += 1

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"[RESULT] Видео обработано ✅")
    print(f"[RESULT] Всего кадров: {frame_count}")
    print(f"[RESULT] Время обработки: {elapsed_time:.2f} сек")
    print(f"[RESULT] Средний FPS: {fps:.2f}")

    print(f"[RESULT] Результат сохранён в папку:")
    print(f"{save_project}/{save_name}")

    # ---------------------------
    # 2. Метрики модели (mAP, precision, recall, F1)
    # ---------------------------
    if not os.path.exists(data_yaml):
        print(f"\n[WARNING] Файл data.yaml не найден: {data_yaml}")
        print("[WARNING] Метрики не посчитаны. Укажи правильный путь к data.yaml.")
        return

    print("\n[INFO] Запускаю валидацию модели на пользовательском датасете...")

    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        verbose=False
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = (2 * precision * recall / (precision + recall + 1e-16))

    print("\n[RESULT] Метрики модели:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\n[INFO] Тестирование завершено!")

if __name__ == "__main__":
    main()