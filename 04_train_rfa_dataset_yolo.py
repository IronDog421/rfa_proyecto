#!/usr/bin/env python3
# ultra-minimal YOLO classification trainer

import argparse

from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO
from src.config.config import rfa_dataset_yolo_train_params

if __name__ == "__main__":
    model = YOLO("yolo11l-seg.pt")
    name="rfa_default_l"

    script_dir = Path(__file__).resolve().parent
    rfa_dataset_dir = script_dir / "datasets" / "rfa_dataset" / "data.yaml"

    start_wall = datetime.now().astimezone().isoformat()
    t0 = perf_counter()

    #nano batch=64
    #small batch=40
    #medium batch=24
    #large batch=18
    #extra batch=12

    try:
        results = model.train(data=rfa_dataset_dir.as_posix(), name=name, device=0, batch=18, exist_ok=False, plots=True, val=True, cache=True, **rfa_dataset_yolo_train_params)
    finally:
        t1 = perf_counter()
        end_wall = datetime.now().astimezone().isoformat()
        elapsed_sec = t1 - t0
        elapsed_hours = elapsed_sec / 3600.0

        save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", script_dir / "runs"))

        report = {
                    "run_name": name,
                    "start_time": start_wall,
                    "end_time": end_wall,
                    "elapsed_seconds": round(elapsed_sec, 3),
                    "elapsed_hours": round(elapsed_hours, 4),
                }
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "train_time.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)