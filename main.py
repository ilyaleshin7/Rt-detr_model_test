import time
import os
import numpy as np
import csv
from ultralytics import RTDETR


# ---------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: безопасно печатает и сохраняет метрики
# ---------------------------
def print_and_save_metrics(metrics, model=None, out_dir="runs/metrics"):
    """
    Эта функция выводит метрики модели (mAP, Precision, Recall, F1)
    и сохраняет их в CSV, если они даны по классам (массивами).
    """

    # создаём папку, куда будут сохраняться результаты
    os.makedirs(out_dir, exist_ok=True)

    box = metrics.box  # тут хранятся метрики детекции

    # Вспомогательная функция: делает среднее, если передан массив
    def mean_or_scalar(x):
        arr = np.array(x)
        if arr.size == 1:      # одно число
            return float(arr), False, arr
        else:                  # несколько чисел (по классам)
            return float(arr.mean()), True, arr

    # ---- выводим главные метрики mAP ----
    print(f"mAP@0.5:      {float(box.map50):.4f}")
    print(f"mAP@0.5:0.95: {float(box.map):.4f}")

    # ---- precision / recall / f1 ----
    p_mean, p_is_arr, p_arr = mean_or_scalar(box.p)
    r_mean, r_is_arr, r_arr = mean_or_scalar(box.r)
    f1_mean, f1_is_arr, f1_arr = mean_or_scalar(box.f1)

    print(f"Precision: {p_mean:.4f}" + (" (среднее по классам)" if p_is_arr else ""))
    print(f"Recall:    {r_mean:.4f}" + (" (среднее по классам)" if r_is_arr else ""))
    print(f"F1-score:  {f1_mean:.4f}" + (" (среднее по классам)" if f1_is_arr else ""))

    # ---- сохраняем per-class метрики, если они есть ----
    if p_is_arr or r_is_arr or f1_is_arr:
        names = None
        if model is not None and hasattr(model, "names"):
            # model.names может быть dict или list
            if isinstance(model.names, dict):
                names = [model.names[i] for i in sorted(model.names)]
            else:
                names = list(model.names)

        def save_csv(filename, header, arr):
            path = os.path.join(out_dir, filename)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, v in enumerate(arr):
                    cls_name = names[i] if names and i < len(names) else f"class_{i}"
                    writer.writerow([cls_name, float(v)])
            print(f"[SAVED] {filename} → {path}")

        if p_is_arr:
            save_csv("per_class_precision.csv", ["class", "precision"], p_arr)
        if r_is_arr:
            save_csv("per_class_recall.csv", ["class", "recall"], r_arr)
        if f1_is_arr:
            save_csv("per_class_f1.csv", ["class", "f1"], f1_arr)

    print(f"\n[INFO] Метрики успешно сохранены в папку: {out_dir}\n")


# ---------------------------
# ОСНОВНОЙ КОД
# ---------------------------
def main():
    # 1. Загружаем обученную модель
    model_path = "best_weights/rtdetr_best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    model = RTDETR(model_path)
    print(f"[INFO] Модель успешно загружена: {model_path}")

    # ---------------------------
    # 2. Анализ видео + измерение FPS
    # ---------------------------
    video_path = "video/video_2025-10-06_20-06-02.mp4"
    print(f"[INFO] Запускаю инференс на видео: {video_path}")

    start_time = time.time()
    frame_count = 0

    # Прогоняем видео через модель
    video_results = model.predict(
        source=video_path,
        save=True,         # сохранить результат в папку
        conf=0.3,          # минимальный уровень уверенности
        stream=True,       # потоковая обработка (для видео)
        project="video_results",
        name="detection"
    )

    # Считаем количество обработанных кадров
    for _ in video_results:
        frame_count += 1

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"\n[RESULT] Видео обработано ✅")
    print(f"[RESULT] Всего кадров: {frame_count}")
    print(f"[RESULT] Время обработки: {elapsed_time:.2f} сек")
    print(f"[RESULT] Средний FPS: {fps:.2f}\n")

    # ---------------------------
    # 3. Валидация модели (оценка качества)
    # ---------------------------
    print("[INFO] Запускаю валидацию модели на твоём датасете...")

    metrics = model.val(data="new_dataset_yolo11/data.yaml")

    print("\n[RESULT] Метрики модели:")
    print_and_save_metrics(metrics, model=model, out_dir="runs/metrics/my_run")

    print("[INFO] Тестирование завершено!")


# ---------------------------
# Точка входа
# ---------------------------
if __name__ == "__main__":
    main()
