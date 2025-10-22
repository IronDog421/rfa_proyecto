import argparse
from io import BytesIO
import logging
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
import kagglehub
import requests

from src.utils.logging import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    script_dir = Path(__file__).resolve().parent
    datasets_dir = script_dir / "datasets"

    datasets_dir.mkdir(parents=False, exist_ok=True)
    logger.info("Datasets folder created at %s", datasets_dir)

    logger.info("Preparing to download DDD and rfa_dataset datasets")

    #################
    ###### DDD ######
    #################

    ddd_path = datasets_dir / "Driver Drowsiness Dataset (DDD)"
    if os.path.exists(ddd_path):
        logger.info("DDD dataset already downloaded at %s", ddd_path)
    else:
        try:
            logger.info("Downloading DDD dataset")
            path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
            path = shutil.copytree(path, datasets_dir, dirs_exist_ok=True)
            logger.info("Successfully downloaded DDD dataset to %s!", path)
        except BaseException:
            logger.exception("Error while downloading the DDD dataset")

    #################
    ## rfa_dataset ##
    #################

    repo_url = "https://github.com/IronDog421/rfa_dataset"
    repo_name = repo_url.split("/")[-1]
    dst = os.path.join(datasets_dir, repo_name)

    if os.path.exists(dst):
        logger.info("rfa_dataset already downloaded at %s", dst)
    else:
        logger.info("Downloading %s from GitHub...", repo_name)
        zip_url = f"{repo_url}/archive/refs/heads/main.zip"
        try:
            response = requests.get(zip_url)
            response.raise_for_status()

            with ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(datasets_dir)

            # Normalize structure (the zip creates a folder with -main suffix)
            extracted_folder = os.path.join(datasets_dir, f"{repo_name}-main")
            shutil.move(extracted_folder, dst)
            logger.info("Dataset successfully downloaded to %s!", dst)
        except Exception:
            logger.exception("Error while downloading the rfa_dataset from GitHub")


if __name__ == "__main__":
    main()
