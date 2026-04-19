from ultralytics import YOLO
import os
import torch

def main():
    # -----------------------------
    # 1. Устройство
    # -----------------------------
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -----------------------------
    model = YOLO("yolo11l.pt")

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
        batch=10,
        device=device,
        workers=8,

        name="yolo11L",

        # --- оптимизация ---
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        patience=30,

        # --- warmup ---
        warmup_epochs=5,
        warmup_momentum=0.8,

        # --- аугментации ---
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,

        fliplr=0.5,
        flipud=0.0,

        scale=0.5,
        translate=0.1,

        mosaic=0.3,
        mixup=0.05,
        copy_paste=0.05,

        close_mosaic=10,

        augment=False,
        verbose=True,
        save_period=10
    )

    print("Обучение завершено.")

if __name__ == "__main__":
    main()