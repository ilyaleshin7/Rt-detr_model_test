from ultralytics import RTDETR

def train_model_full():
    # Загружаем лёгкую версию модели
    model = RTDETR("rtdetr-l.pt")

    # --- Обучение ---
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=512,
        batch=4,
        workers=2,
        device=0,
        optimizer="AdamW",
        lr0=0.0003,
        deterministic=False,
        val=True,                     # считать метрики после каждой эпохи
        project="runs/train",
        name="rtdetr_s_aug_3050ti",
        plots=True,
        verbose=True,
        save_period=5,
        # --- Аугментации ---
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
        translate=0.1,
        erasing=0.4,
        auto_augment="randaugment"
    )

    # --- Автоматическая финальная валидация ---
    print("\n[INFO] Обучение завершено. Запускаю финальную оценку метрик...")
    metrics = model.val(data="dataset/data.yaml")

    print("\n[RESULT] Итоговые метрики:")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.p:.4f}")
    print(f"Recall:       {metrics.box.r:.4f}")
    print(f"F1-score:     {metrics.box.f1:.4f}")

    print("\n[INFO] Результаты обучения сохранены в:")
    print(f" {results.save_dir}")

if __name__ == "__main__":
    train_model_full()