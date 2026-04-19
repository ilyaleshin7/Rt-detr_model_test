from ultralytics import RTDETR
import os
import torch

def main():
    # -----------------------------
    # 1. Устройство
    # -----------------------------
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -----------------------------
    # 2. МОДЕЛЬ RT-DETR
    # -----------------------------
    model = RTDETR("rtdetr-l.pt")

    # -----------------------------
    # 3. DATA
    # -----------------------------
    data_path = "../new_dataset_yolo11/data.yaml"

    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        return

    # -----------------------------
    # 4. TRAIN
    # -----------------------------
    results = model.train(
        data=data_path,

        # --- базовые ---
        epochs=150,
        imgsz=640,
        batch=2,
        device=device,
        workers=4,
        name="rtdetr",

        # --- оптимизация ---
        lr0=0.0005,
        lrf=0.01,
        patience=20,

        # --- важно ---
        cos_lr=False,
        warmup_epochs=5,

        # --- аугментации (мягче, чем у YOLO) ---
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.1,

        fliplr=0.5,
        scale=0.3,
        translate=0.1,

        mosaic=0.0,      # отключаем
        mixup=0.0,       # отключаем
        copy_paste=0.0,  # отключаем

        # --- прочее ---
        cache=False,
        augment=False,
        verbose=True,
        save_period=10,
        max_det= 100,
        amp=True
    )

    print("Обучение завершено.")

if __name__ == "__main__":
    main()