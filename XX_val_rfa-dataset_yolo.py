
#!/usr/bin/env python3
# ultra-minimal YOLO classification trainer

import argparse

from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from ultralytics import YOLO
from src.config.config import ddd_yolo_train_params

if __name__ == "__main__":
    model = YOLO("runs/segment/rfa_default_s/weights/best.pt")
    name="rfa_default_s"

    script_dir = Path(__file__).resolve().parent
    rfa_dataset_dir = script_dir / "datasets" / "rfa_dataset" / "data.yaml"

    start_wall = datetime.now().astimezone().isoformat()
    t0 = perf_counter()

    #nano batch=64
    #small batch=40
    #medium batch=24
    #large batch=18
    #extra batch=12

    results = model.val(data=rfa_dataset_dir.as_posix(), name=name, device=0, batch=64, exist_ok=False, plots=True, val=True, cache=True, **ddd_yolo_train_params)
