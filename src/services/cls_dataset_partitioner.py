#!/usr/bin/env python3
"""
Stratified Dataset Splitter for Ultralytics YOLO Classification

A production-quality CLI tool that automatically performs stratified splitting
of folder-per-class image datasets into train/val/test splits compatible with
Ultralytics YOLO Classification format.

Author: Senior Python Engineer
License: MIT
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import random
import json
from datetime import datetime
from dataclasses import dataclass, field

from src.utils.logging import setup_logging

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

@dataclass
class ClassifiedDatasetPartitioner:
    """Handles stratified splitting of image classification datasets."""
    input_dir: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    min_samples_per_class: int = 3
    copy_files: bool = True

    # Internal fields (not provided by user)
    logger: logging.Logger = field(init=False, repr=False)
    stats: Dict[str, Dict[str, int]] = field(init=False, default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        """
        Post-initialization: normalize paths, validate ratios, setup logger and RNG,
        and ensure stats is a defaultdict.
        """
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        random.seed(self.seed)

        # Ensure stats is a defaultdict
        if not isinstance(self.stats, defaultdict):
            self.stats = defaultdict(dict)

    def validate_input_structure(self) -> Dict[str, List[Path]]:
        """
        Validate input directory structure and collect image files.
        """
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")
        
        class_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in: {self.input_dir}")
        
        class_files: Dict[str, List[Path]] = {}
        total_images = 0
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            if class_name.startswith('.') or class_name.lower() in {'__pycache__', '.git'}:
                continue
                
            image_files = []
            for file_path in class_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in SUPPORTED_EXTENSIONS):
                    image_files.append(file_path)
            
            if len(image_files) < self.min_samples_per_class:
                self.logger.warning(
                    f"Class '{class_name}' has only {len(image_files)} samples "
                    f"(minimum required: {self.min_samples_per_class}). Skipping."
                )
                continue
            
            class_files[class_name] = image_files
            total_images += len(image_files)
            self.logger.info(f"Class '{class_name}': {len(image_files)} images")
        
        if not class_files:
            raise ValueError("No valid classes found with sufficient samples")
        
        self.logger.info(f"Found {len(class_files)} classes with {total_images} total images")
        return class_files
    
    def stratified_split(self, class_files: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[Path]]]:
        """
        Perform stratified split maintaining class distribution across splits.
        """
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for class_name, files in class_files.items():
            # Shuffle files for this class
            files_copy = files.copy()
            random.shuffle(files_copy)
            
            n_files = len(files_copy)
            n_train = max(1, int(n_files * self.train_ratio))
            n_val = max(1, int(n_files * self.val_ratio))
            n_test = n_files - n_train - n_val
            
            # Ensure at least one sample in test if we have enough files
            if n_test < 1 and n_files >= 3:
                n_test = 1
                n_val = max(1, n_files - n_train - n_test)
            
            # Split files
            train_files = files_copy[:n_train]
            val_files = files_copy[n_train:n_train + n_val]
            test_files = files_copy[n_train + n_val:]
            
            splits['train'][class_name] = train_files
            splits['val'][class_name] = val_files
            splits['test'][class_name] = test_files
            
            # Store statistics
            self.stats[class_name] = {
                'total': n_files,
                'train': len(train_files),
                'val': len(val_files),
                'test': len(test_files)
            }
            
            self.logger.info(
                f"Class '{class_name}' split: "
                f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
            )
        
        return splits
    
    def create_output_structure(self, splits: Dict[str, Dict[str, List[Path]]]) -> None:
        """
        Create the output directory structure and copy/link files.
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split_name in ['train', 'val', 'test']:
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create class directories within each split
            for class_name in splits[split_name].keys():
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
        
        # Copy or link files
        total_copied = 0
        for split_name, split_classes in splits.items():
            split_dir = self.output_dir / split_name
            
            for class_name, files in split_classes.items():
                class_dir = split_dir / class_name
                
                for src_file in files:
                    dst_file = class_dir / src_file.name
                    
                    # Handle filename conflicts
                    if dst_file.exists():
                        stem = src_file.stem
                        suffix = src_file.suffix
                        counter = 1
                        while dst_file.exists():
                            dst_file = class_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    try:
                        if self.copy_files:
                            shutil.copy2(src_file, dst_file)
                        else:
                            dst_file.symlink_to(src_file.resolve())
                        total_copied += 1
                    except Exception as e:
                        self.logger.error(f"Failed to process {src_file}: {e}")
        
        self.logger.info(f"Successfully processed {total_copied} files")


    def generate_summary_report(self, class_files: Dict[str, List[Path]]) -> None:
        """Generate and save a summary report of the split operation."""
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'split_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio,
                    'test': self.test_ratio
                },
                'random_seed': self.seed,
                'copy_files': self.copy_files
            },
            'summary': {
                'total_classes': len(class_files),
                'total_images': sum(len(files) for files in class_files.values())
            },
            'class_statistics': dict(self.stats),
            'classes': sorted(class_files.keys()),
        }
        
        # Calculate totals per split
        totals = {'train': 0, 'val': 0, 'test': 0}
        for stats in self.stats.values():
            for split in totals:
                totals[split] += stats[split]
        
        report['split_totals'] = totals
        
        # Save JSON report
        report_path = self.output_dir / 'split_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("DATASET SPLIT SUMMARY")
        print("="*60)
        print(f"Input Directory: {self.input_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Total Classes: {report['summary']['total_classes']}")
        print(f"Classes: {', '.join(report['classes'])}")
        print(f"Total Images: {report['summary']['total_images']}")
        print(f"\nSplit Distribution:")
        print(f"  Train: {totals['train']} images ({totals['train']/report['summary']['total_images']:.1%})")
        print(f"  Val:   {totals['val']} images ({totals['val']/report['summary']['total_images']:.1%})")
        print(f"  Test:  {totals['test']} images ({totals['test']/report['summary']['total_images']:.1%})")
        print("="*60)
    
    def run(self) -> None:
        """Execute the complete dataset splitting process."""
        try:
            self.logger.info("Starting dataset splitting process...")
            
            class_files = self.validate_input_structure()
            
            splits = self.stratified_split(class_files)
            
            self.create_output_structure(splits)
            
            self.generate_summary_report(class_files)
            
            self.logger.info("Dataset splitting completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Dataset splitting failed: {e}")
            raise





def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified Classification Dataset Splitter for Ultralytics YOLO Classification",
        epilog="""
        Examples:
        # Basic usage with default 70/15/15 split
        python split_dataset.py /path/to/input /path/to/output
        
        # Custom split ratios
        python split_dataset.py /path/to/input /path/to/output --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        
        # Use symlinks instead of copying (faster, less disk space)
        python split_dataset.py /path/to/input /path/to/output --symlink
        
        # Custom random seed for reproducibility
        python split_dataset.py /path/to/input /path/to/output --seed 123
                """
    )
    
    parser.add_argument('input_dir', type=Path, help='Input directory containing class folders with images')
    parser.add_argument('output_dir', type=Path, help='Output directory for the split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Proportion of data for training set (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Proportion of data for validation set (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Proportion of data for test set (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--min-samples', type=int, default=3, help='Minimum samples required per class (default: 3)')
    parser.add_argument('--symlink', action='store_true', help='Create symlinks instead of copying files (saves disk space)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it exists')
    
    return parser.parse_args()

def main() -> int:

    try:
        args = parse_arguments()
        
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        if not args.input_dir.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1
        
        if args.output_dir.exists() and not args.overwrite:
            logger.error(
                f"Output directory already exists: {args.output_dir}\n"
                "Use --overwrite to overwrite existing directory"
            )
            return 1
        
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.error(f"Split ratios must sum to 1.0, got {total_ratio}")
            return 1
        
        splitter = ClassifiedDatasetPartitioner(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_samples_per_class=args.min_samples,
            copy_files=not args.symlink
        )
        
        splitter.run()
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())