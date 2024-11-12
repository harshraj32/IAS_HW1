import torch
import torchvision
import numpy as np
from PIL import Image
import os
import csv
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import datetime
import time
import math
import sys
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.transforms import transforms as T
import utils
import random
from myevaluator import CocoEvaluator , get_coco_api_from_dataset


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        return image.float() / 255.0, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            bbox = target["boxes"]
            
            # Flip boxes: [x1, y1, x2, y2] -> [W-x2, y1, W-x1, y2]
            W = image.shape[-1]
            bbox[:, [0, 2]] = W - bbox[:, [2, 0]]
            target["boxes"] = bbox
            
        return image, target

def create_train_val_splits(root_dir: str, train_ratio: float = 0.8):
    """Create train and validation split files"""
    root_path = Path(root_dir)
    image_dir = root_path / "training" / "image_2"  # Removed 'raw' from path
    imagesets_dir = root_path / "ImageSets"
    
    # Create ImageSets directory if it doesn't exist
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(image_dir.glob("*.png")))
    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")
        
    # Get image IDs (without extension)
    image_ids = [f.stem for f in image_files]
    
    # Randomly shuffle the IDs
    random.shuffle(image_ids)
    
    # Split into train and validation
    num_train = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:num_train]
    val_ids = image_ids[num_train:]
    
    # Write splits to files
    with open(imagesets_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids))
    
    with open(imagesets_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids))
    
    print(f"Created train split with {len(train_ids)} images")
    print(f"Created val split with {len(val_ids)} images")


class KITTIDetectionDataset(Dataset):
    """
    Custom KITTI dataset for object detection
    """
    def __init__(self, 
                 root: str,
                 split: str = 'train',
                 subset_fraction: float = 1.0,
                 transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Define class names and mapping
        self.classes = ['__background__', 'Car', 'Van', 'Truck', 'Pedestrian', 
                       'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Setup paths
        self.images_dir = self.root / 'training' / 'image_2'
        self.labels_dir = self.root / 'training' / 'label_2'
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise RuntimeError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise RuntimeError(f"Labels directory not found: {self.labels_dir}")
            
        # Get image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")
            
        # Create train/val split
        if split == 'train':
            num_train = int(len(self.image_files) * 0.8)
            self.image_files = self.image_files[:num_train]
        elif split == 'val':
            num_train = int(len(self.image_files) * 0.8)
            self.image_files = self.image_files[num_train:]
            
        # Apply subset fraction if needed
        if subset_fraction < 1.0:
            num_samples = int(len(self.image_files) * subset_fraction)
            self.image_files = random.sample(self.image_files, num_samples)
            print(f"Using {len(self.image_files)} images ({subset_fraction*100}%)")
        
        # Create image id mapping
        self.img_id_map = {f.stem: idx for idx, f in enumerate(self.image_files)}

    def __len__(self) -> int:
        return len(self.image_files)

    def get_image(self, image_file: Path) -> Image.Image:
        return Image.open(image_file).convert('RGB')

    def get_target(self, image_file: Path) -> dict:
        # Get corresponding label file
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        boxes = []
        labels = []
        
        with open(label_file) as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                class_name = line[0]
                if class_name == 'DontCare':
                    continue
                    
                # Get bounding box coordinates
                bbox = [float(x) for x in line[4:8]]
                if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    continue
                    
                boxes.append(bbox)
                labels.append(self.class_to_idx[class_name])

        # Convert to tensor format
        target = {}
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            # Use mapped image ID instead of file stem
            target["image_id"] = torch.tensor([self.img_id_map[image_file.stem]])
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * \
                            (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([self.img_id_map[image_file.stem]])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_file = self.image_files[index]
        image = self.get_image(image_file)
        target = self.get_target(image_file)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
def get_transform(train):
    transforms = []
    # Convert PIL image to tensor and normalize
    transforms.append(ToTensor())
    
    if train:
        # Add training transforms
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)


def train_kitti_detection(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("Creating datasets...")
    # Create datasets
    dataset = KITTIDetectionDataset(
        root=args.data_path,
        split='train',
        subset_fraction=0.01,
        transform=get_transform(train=True)
    )
    
    dataset_test = KITTIDetectionDataset(
        root=args.data_path,
        split='val',
        transform=get_transform(train=False)
    )
    
    print(f"Number of training images: {len(dataset)}")
    print(f"Number of validation images: {len(dataset_test)}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=utils.collate_fn
    )
    
    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=utils.collate_fn
    )
    
    # Create model
    print("Creating model...")
    num_classes = len(dataset.classes)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        num_classes=num_classes,
        pretrained_backbone=True
    )
    model.to(device)
    
    # Setup optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_steps,
        gamma=args.lr_gamma
    )
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate
        print(f"Evaluating epoch {epoch}...")
        evaluate(model, test_loader, device=device)
        
        # Save checkpoint
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            checkpoint_path = Path(args.output_dir) / f"model_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    return model


def evaluate(model, data_loader, device):
    model.eval()
    
    # Initialize COCO evaluator
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            
            # Get model predictions
            outputs = model(images)
            
            # Process outputs
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            
            # Create results dictionary using consistent image IDs
            res = {}
            for target, output in zip(targets, outputs):
                img_id = target["image_id"].item()
                res[img_id] = output
            
            # Update evaluator
            try:
                coco_evaluator.update(res)
            except AssertionError as e:
                print(f"Warning: Evaluation error - {str(e)}")
                print(f"Target image IDs: {[t['image_id'].item() for t in targets]}")
                print(f"Output image IDs: {list(res.keys())}")
                continue
            
    # Compute and print metrics
    print("Computing evaluation metrics...")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return coco_evaluator

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='KITTI Object Detection Training')
    parser.add_argument('--data-path', default='/path/to/kitti', help='dataset root path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=26, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--output-dir', default='output', help='path to save outputs')
    
    args = parser.parse_args()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
    create_train_val_splits(args.data_path)
    model = train_kitti_detection(args)