import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import wget


class KITTIDownloader:
    """
    Handles downloading and extracting KITTI dataset for object detection
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "raw"
        self.imagesets_dir = self.root_dir / "ImageSets"

        # Official KITTI URLs
        self.urls = {
            "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
            "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
            "devkit_object.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip",
        }

    def download_file(self, url: str, filename: str) -> None:
        """Download a file from URL if it doesn't exist"""
        filepath = self.root_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            wget.download(url, str(filepath))
            print(f"\nFinished downloading {filename}")
        else:
            print(f"File {filename} already exists, skipping download")

    def extract_zip(self, filename: str, extract_dir: Optional[str] = None) -> None:
        """Extract a zip file"""
        filepath = self.root_dir / filename
        extract_path = (
            self.root_dir if extract_dir is None else self.root_dir / extract_dir
        )

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Finished extracting {filename}")

    def prepare_directory_structure(self) -> None:
        """Create necessary directories"""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.imagesets_dir.mkdir(exist_ok=True)

    def create_train_val_split(self, train_ratio: float = 0.8) -> None:
        """Create train/val split files"""
        # Get all image files
        image_dir = self.raw_dir / "training" / "image_2"
        all_images = sorted([f.stem for f in image_dir.glob("*.png")])

        # Calculate split
        num_train = int(len(all_images) * train_ratio)
        train_images = all_images[:num_train]
        val_images = all_images[num_train:]

        # Write split files
        with open(self.imagesets_dir / "train.txt", "w") as f:
            f.write("\n".join(train_images))

        with open(self.imagesets_dir / "val.txt", "w") as f:
            f.write("\n".join(val_images))

        print(f"Created train split with {len(train_images)} images")
        print(f"Created val split with {len(val_images)} images")

    def cleanup_downloads(self) -> None:
        """Remove downloaded zip files"""
        for filename in self.urls.keys():
            filepath = self.root_dir / filename
            if filepath.exists():
                filepath.unlink()
                print(f"Removed {filename}")

    def verify_dataset(self) -> bool:
        """Verify that the dataset is properly downloaded and extracted"""
        required_dirs = [
            self.raw_dir / "training" / "image_2",
            self.raw_dir / "training" / "label_2",
            self.raw_dir / "testing" / "image_2",
            self.imagesets_dir,
        ]

        for directory in required_dirs:
            if not directory.exists():
                print(f"Missing directory: {directory}")
                return False

            # Check if directories contain files
            if len(list(directory.glob("*"))) == 0:
                print(f"Directory is empty: {directory}")
                return False

        return True

    def download_and_prepare(self) -> None:
        """Main method to download and prepare the dataset"""
        print("Starting KITTI dataset download and preparation...")

        # Create directories
        self.prepare_directory_structure()

        # Download files
        for filename, url in self.urls.items():
            self.download_file(url, filename)
            self.extract_zip(filename)

        # Create train/val split
        self.create_train_val_split()

        # Verify dataset
        if self.verify_dataset():
            print("Dataset successfully prepared!")

            # Optional: cleanup downloaded zip files
            self.cleanup_downloads()
        else:
            print("Dataset preparation failed!")


def download_kitti():
    """Command line interface for downloading KITTI dataset"""
    parser = argparse.ArgumentParser(description="Download and prepare KITTI dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory to store the dataset",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )
    args = parser.parse_args()

    downloader = KITTIDownloader(args.data_dir)
    downloader.download_and_prepare()

    if not args.keep_zips:
        downloader.cleanup_downloads()


def get_dataset_stats():
    """Print dataset statistics after download"""
    parser = argparse.ArgumentParser(description="Get KITTI dataset statistics")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Root directory of the dataset"
    )
    args = parser.parse_args()

    root_dir = Path(args.data_dir)

    # Count images
    train_images = len(list((root_dir / "raw" / "training" / "image_2").glob("*.png")))
    test_images = len(list((root_dir / "raw" / "testing" / "image_2").glob("*.png")))

    # Count labels
    label_files = len(list((root_dir / "raw" / "training" / "label_2").glob("*.txt")))

    print("\nKITTI Dataset Statistics:")
    print(f"Training Images: {train_images}")
    print(f"Testing Images: {test_images}")
    print(f"Label Files: {label_files}")


if __name__ == "__main__":
    # Example usage
    download_kitti()
    get_dataset_stats()
