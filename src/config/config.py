rfa_dataset_yolo_train_params = {
    # --- Optimization / LR schedule ---
    "lr0": 0.01,             # initial learning rate
    "lrf": 0.01,             # final LR as fraction of lr0 (lr_final = lr0 * lrf)
    "momentum": 0.937,       # SGD momentum / Adam beta1
    "weight_decay": 0.0005,  # L2 regularization
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "patience": 20,
    "epochs": 100,

    # --- Loss weights ---
    "box": 7.5,              # bbox loss weight
    "cls": 0.5,              # class loss weight
    "dfl": 1.5,              # Distribution Focal Loss weight
    "nbs": 64,               # nominal batch size used to normalize loss

    # --- Augmentation (train) ---
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,       # (segmentation only)

    # --- Segmentation-only ---
    "overlap_mask": True,
    "mask_ratio": 4,
}


ddd_ratios = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

ddd_yolo_train_params = {
    # --- Optimization / LR schedule ---
    "lr0": 0.01,             # initial learning rate
    "lrf": 0.01,             # final LR as fraction of lr0 (lr_final = lr0 * lrf)
    "momentum": 0.937,       # SGD momentum / Adam beta1
    "weight_decay": 0.0005,  # L2 regularization
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "patience": 20,
    "epochs": 100,

    # --- Loss weights ---
    "box": 7.5,              # bbox loss weight
    "cls": 0.5,              # class loss weight
    "dfl": 1.5,              # Distribution Focal Loss weight
    "nbs": 64,               # nominal batch size used to normalize loss

    # --- Augmentation (train) ---
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,

    # --- Classification-only (ignored for detect/segment) ---
    "dropout": 0.0,
    "auto_augment": "randaugment",
    "erasing": 0.4,
}
