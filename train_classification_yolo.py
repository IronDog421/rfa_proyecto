#!/usr/bin/env python3
# ultra-minimal YOLO classification trainer

import argparse
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11x-cls.pt")

    results = model.train(data="/home/ruiz/RFA_wsl/Transformed_Datasets/DDD/", device=0, epochs=100, batch=0.85, exist_ok=False, plots=True, val=True, cache=True, patience=20)