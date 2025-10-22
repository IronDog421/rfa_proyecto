import argparse
from io import BytesIO
import logging
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
import kagglehub
import requests

from src.services.cls_dataset_partitioner import ClassifiedDatasetPartitioner
from src.config.config import ddd_ratios
from src.utils.logging import setup_logging

def main():
    script_dir = Path(__file__).resolve().parent
    ddd_dir = script_dir / "datasets" / "Driver Drowsiness Dataset (DDD)"

    transformed_datasets_dir = script_dir / "transformed_datasets" / "DDD"

    transformed_datasets_dir.mkdir(parents=True, exist_ok=True)

    splitter = ClassifiedDatasetPartitioner(
        input_dir=ddd_dir,
        output_dir=transformed_datasets_dir,
        train_ratio=ddd_ratios["train"],
        val_ratio=ddd_ratios["val"],
        test_ratio=ddd_ratios["test"],
    )
    
    splitter.run()


if __name__ == "__main__":
    main()
