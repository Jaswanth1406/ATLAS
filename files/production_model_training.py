# 🚀 ATLAS — PRODUCTION MODEL TRAINING
# Complete pipeline to train a deployable road segmentation model
# Works on Google Colab or any local machine with a GPU

"""
PRODUCTION-READY MODEL TRAINING PIPELINE

Features:
- UNet architecture for semantic segmentation
- Mixed precision training (faster)
- Model checkpointing (best model saved)
- Data augmentation
- Learning rate scheduling
- Early stopping
- Model export for deployment (ONNX + TorchScript)

Usage:
    # Google Colab:
    %run production_model_training.py

    # Local / CLI:
    python production_model_training.py \
        --data_dir ./datasets \
        --output_dir ./output \
        --epochs 50 \
        --batch_size 16

    Dataset structure expected:
        <data_dir>/
        ├── train/
        │   ├── img/     ← training images (.png/.jpg)
        │   └── label/   ← corresponding label masks
        └── val/
            ├── img/     ← validation images
            └── label/   ← corresponding label masks

Output: Trained model ready for FastAPI backend
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

print("📦 Installing dependencies...")

import subprocess
import sys

packages = [
    "torch",
    "torchvision",
    "albumentations",
    "segmentation-models-pytorch",
    "onnx",
    "onnxruntime",
]

for package in packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("✅ All dependencies installed!")

# ============================================================================
# CELL 2: Imports
# ============================================================================

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import zipfile
from collections import defaultdict
from pathlib import Path

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# Torchvision
import torchvision.transforms as T

# Segmentation Models
import segmentation_models_pytorch as smp

# Albumentations for augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Google Colab support (optional)
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

# W&B (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

print("✅ Imports complete!")

# ============================================================================
# CELL 3: Configuration
# ============================================================================

class Config:
    """Production model configuration — all paths are configurable."""

    # Paths (set via CLI args or defaults)
    DATA_DIR = "./datasets"         # Root folder containing train/ and val/
    OUTPUT_DIR = "./output"         # Where models + exports are saved
    ZIP_PATH = None                 # Optional: path to a .zip dataset to extract

    # Model architecture
    MODEL_NAME = "UNet"
    ENCODER = "resnet34"            # resnet34, efficientnet-b0, mobilenet_v2
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1                     # Binary segmentation (road vs non-road)

    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Image size
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    # Road class IDs (pixel values in label masks that represent "road")
    # Adjust this for your dataset!
    ROAD_CLASSES = [70, 89, 69, 71, 72, 68, 90, 105, 106, 117, 118, 138]

    # Training settings
    USE_MIXED_PRECISION = True      # Faster training on modern GPUs
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # Early stopping
    PATIENCE = 10                   # Stop if no improvement for N epochs
    MIN_DELTA = 0.001               # Minimum improvement threshold

    # Model saving
    SAVE_BEST_ONLY = True
    SAVE_FREQUENCY = 5              # Save checkpoint every N epochs

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Team info
    TEAM = ["Divya R", "Haripriya K", "Jaswanth Prasanna V"]

    @classmethod
    def from_args(cls, args):
        """Update config from CLI arguments."""
        if args.data_dir:
            cls.DATA_DIR = args.data_dir
        if args.output_dir:
            cls.OUTPUT_DIR = args.output_dir
        if args.zip_path:
            cls.ZIP_PATH = args.zip_path
        if args.epochs:
            cls.NUM_EPOCHS = args.epochs
        if args.batch_size:
            cls.BATCH_SIZE = args.batch_size
        if args.lr:
            cls.LEARNING_RATE = args.lr
        if args.encoder:
            cls.ENCODER = args.encoder
        if args.img_size:
            cls.IMG_HEIGHT = args.img_size
            cls.IMG_WIDTH = args.img_size

    @classmethod
    def setup_dirs(cls):
        """Create output directories."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "exports"), exist_ok=True)


# ============================================================================
# CELL 4: Dataset Class
# ============================================================================

class RoadSegmentationDataset(Dataset):
    """Production dataset with augmentation"""

    def __init__(self, img_dir, label_dir, transform=None, is_train=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train

        # Get image files
        self.images = sorted([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        print(f"   Loaded {len(self.images)} images from {img_dir}")

    def __len__(self):
        return len(self.images)

    def load_and_convert_label(self, label_path):
        """Convert multi-class semantic segmentation to binary road mask"""
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label is None:
            label = cv2.imread(label_path)
            if label is not None:
                label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            else:
                return None

        # Convert to binary: road = 1, non-road = 0
        road_mask = np.zeros_like(label, dtype=np.float32)
        for road_class in Config.ROAD_CLASSES:
            road_mask[label == road_class] = 1.0

        return road_mask

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label_path = os.path.join(self.label_dir, self.images[idx])
        mask = self.load_and_convert_label(label_path)

        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Ensure mask is [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask

# ============================================================================
# CELL 5: Data Augmentation
# ============================================================================

def get_train_transform():
    """Training augmentation pipeline"""
    return A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    """Validation/test transform (no augmentation)"""
    return A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

print("✅ Augmentation pipelines defined!")

# ============================================================================
# CELL 6: Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

print("✅ Loss functions defined!")

# ============================================================================
# CELL 7: Metrics
# ============================================================================

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU metric"""
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    return dice.item()

print("✅ Metrics defined!")

# ============================================================================
# CELL 8: Model Definition
# ============================================================================

def create_model():
    """Create segmentation model"""

    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
        activation=None,  # We'll use sigmoid in loss
    )

    return model

print("✅ Model architecture defined!")

# ============================================================================
# CELL 9: Training Loop
# ============================================================================

class Trainer:
    """Production model trainer"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Mixed precision
        self.scaler = GradScaler() if Config.USE_MIXED_PRECISION else None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []

        # Best model tracking
        self.best_val_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # Timestamps
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")

        for images, masks in pbar:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            self.optimizer.zero_grad()

            # Mixed precision training
            if Config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Metrics
                iou = calculate_iou(outputs, masks)
                dice = calculate_dice(outputs, masks)

                epoch_loss += loss.item()
                epoch_iou += iou
                epoch_dice += dice

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })

        avg_loss = epoch_loss / len(self.val_loader)
        avg_iou = epoch_iou / len(self.val_loader)
        avg_dice = epoch_dice / len(self.val_loader)

        self.val_losses.append(avg_loss)
        self.val_ious.append(avg_iou)
        self.val_dices.append(avg_dice)

        return avg_loss, avg_iou, avg_dice

    def save_checkpoint(self, epoch, iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iou': iou,
            'config': {
                'model_name': Config.MODEL_NAME,
                'encoder': Config.ENCODER,
                'img_size': (Config.IMG_HEIGHT, Config.IMG_WIDTH),
                'road_classes': Config.ROAD_CLASSES,
            },
            'team': Config.TEAM,
        }

        if is_best:
            path = os.path.join(Config.OUTPUT_DIR, "checkpoints", "best_model.pth")
            torch.save(checkpoint, path)
            print(f"   💾 Best model saved! IoU: {iou:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            path = os.path.join(Config.OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, path)

    def early_stopping_check(self, val_iou):
        """Check if training should stop"""
        if val_iou > self.best_val_iou + Config.MIN_DELTA:
            self.best_val_iou = val_iou
            self.patience_counter = 0
            return False, True  # Continue, is_best
        else:
            self.patience_counter += 1
            if self.patience_counter >= Config.PATIENCE:
                return True, False  # Stop, not_best
            return False, False  # Continue, not_best

    def train(self):
        """Complete training loop"""
        print("\n" + "="*80)
        print("🚀 STARTING PRODUCTION MODEL TRAINING")
        print("="*80)
        print(f"   Model: {Config.MODEL_NAME} ({Config.ENCODER})")
        print(f"   Device: {Config.DEVICE}")
        print(f"   Batch size: {Config.BATCH_SIZE}")
        print(f"   Epochs: {Config.NUM_EPOCHS}")
        print(f"   Learning rate: {Config.LEARNING_RATE}")
        print(f"   Data: {Config.DATA_DIR}")
        print(f"   Output: {Config.OUTPUT_DIR}")
        print("="*80 + "\n")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_iou, val_dice = self.validate_epoch(epoch)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val IoU: {val_iou:.4f}")
            print(f"   Val Dice: {val_dice:.4f}")

            # Early stopping check
            should_stop, is_best = self.early_stopping_check(val_iou)

            # Save checkpoint
            self.save_checkpoint(epoch, val_iou, is_best=is_best)

            if should_stop:
                print(f"\n⚠️ Early stopping triggered! No improvement for {Config.PATIENCE} epochs.")
                print(f"   Best IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch + 1}")
                break

            if is_best:
                self.best_epoch = epoch

            print()

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"   Best Val IoU: {self.best_val_iou:.4f}")
        print(f"   Best Epoch: {self.best_epoch + 1}")
        print(f"   Training time: {datetime.now() - self.start_time}")
        print("="*80)

        return self.best_val_iou

print("✅ Trainer class defined!")

# ============================================================================
# CELL 10: Model Export for Production
# ============================================================================

def export_model_for_production(model, save_dir):
    """Export model in multiple formats for production deployment"""

    model.eval()
    model.to('cpu')

    # Create dummy input
    dummy_input = torch.randn(1, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)

    print("\n" + "="*80)
    print("📦 EXPORTING MODEL FOR PRODUCTION")
    print("="*80)

    # 1. PyTorch (.pth) - Already saved
    print("✅ PyTorch checkpoint: best_model.pth")

    # 2. TorchScript (.pt) - For PyTorch deployment
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = os.path.join(save_dir, "exports", "model.pt")
        traced_model.save(torchscript_path)
        print(f"✅ TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"⚠️ TorchScript export failed: {e}")

    # 3. ONNX (.onnx) - For cross-platform deployment
    try:
        onnx_path = os.path.join(save_dir, "exports", "model.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ ONNX: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")

    # 4. Save config for deployment
    config_dict = {
        'model_name': Config.MODEL_NAME,
        'encoder': Config.ENCODER,
        'input_shape': [Config.IMG_HEIGHT, Config.IMG_WIDTH],
        'road_classes': Config.ROAD_CLASSES,
        'num_classes': Config.CLASSES,
        'team': Config.TEAM,
        'trained_date': datetime.now().isoformat(),
    }

    config_path = os.path.join(save_dir, "exports", "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config: {config_path}")

    print("="*80)
    print("✅ All export formats ready for deployment!")
    print("="*80)

# ============================================================================
# CELL 11: Main Execution
# ============================================================================

def main():
    """Main training pipeline"""

    # Setup directories
    Config.setup_dirs()

    # If a zip file is provided, extract it
    if Config.ZIP_PATH and os.path.exists(Config.ZIP_PATH):
        print(f"📦 Extracting dataset from {Config.ZIP_PATH}...")
        with zipfile.ZipFile(Config.ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(Config.DATA_DIR)
        print("✅ Dataset extracted!")

    # Dataset paths
    train_img_dir = os.path.join(Config.DATA_DIR, "train", "img")
    train_label_dir = os.path.join(Config.DATA_DIR, "train", "label")
    val_img_dir = os.path.join(Config.DATA_DIR, "val", "img")
    val_label_dir = os.path.join(Config.DATA_DIR, "val", "label")

    # Validate paths exist
    for d, name in [(train_img_dir, "train/img"), (train_label_dir, "train/label"),
                     (val_img_dir, "val/img"), (val_label_dir, "val/label")]:
        if not os.path.isdir(d):
            print(f"❌ Missing directory: {d}")
            print(f"\n💡 Expected dataset structure:")
            print(f"   {Config.DATA_DIR}/")
            print(f"   ├── train/")
            print(f"   │   ├── img/    ← training images")
            print(f"   │   └── label/  ← training labels")
            print(f"   └── val/")
            print(f"       ├── img/    ← validation images")
            print(f"       └── label/  ← validation labels")
            return None, 0.0

    print("\n📂 Loading datasets...")
    train_dataset = RoadSegmentationDataset(
        train_img_dir, train_label_dir,
        transform=get_train_transform(),
        is_train=True
    )

    val_dataset = RoadSegmentationDataset(
        val_img_dir, val_label_dir,
        transform=get_val_transform(),
        is_train=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")

    # Create model
    print("\n🏗️ Building model...")
    model = create_model()
    print(f"✅ Model created: {Config.MODEL_NAME} with {Config.ENCODER}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)
    best_iou = trainer.train()

    # Export for production
    print("\n📦 Exporting model for production...")
    export_model_for_production(model, Config.OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("🎉 PRODUCTION MODEL READY!")
    print("="*80)
    print(f"   Best Validation IoU: {best_iou:.4f}")
    print(f"\n📁 Output files in: {Config.OUTPUT_DIR}")

    exports_dir = os.path.join(Config.OUTPUT_DIR, "exports")
    for name in ["model.pt", "model.onnx", "model_config.json"]:
        path = os.path.join(exports_dir, name)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"   ✅ {name} ({size_mb:.1f} MB)")

    ckpt_path = os.path.join(Config.OUTPUT_DIR, "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024*1024)
        print(f"   ✅ best_model.pth ({size_mb:.1f} MB)")

    print("="*80)
    print("\n💡 Next steps:")
    print(f"   1. Copy {exports_dir}/model.pt → backend/models/model.pt")
    print(f"   2. Copy {exports_dir}/model_config.json → backend/models/model_config.json")
    print(f"   3. Start the backend: uvicorn main:app --reload --port 8000")

    return model, best_iou


# ============================================================================
# CLI argument parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ATLAS — Production Road Segmentation Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (expects ./datasets/)
  python production_model_training.py

  # Specify custom paths
  python production_model_training.py --data_dir /path/to/data --output_dir /path/to/output

  # Extract from zip first
  python production_model_training.py --zip_path /path/to/dataset.zip --data_dir ./extracted

  # Customize training
  python production_model_training.py --epochs 100 --batch_size 8 --lr 0.0005 --encoder efficientnet-b0
        """,
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Root dataset directory (contains train/ and val/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for models and exports")
    parser.add_argument("--zip_path", type=str, default=None, help="Optional: path to dataset .zip to extract")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4)")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder backbone: resnet34, efficientnet-b0, mobilenet_v2")
    parser.add_argument("--img_size", type=int, default=None, help="Image size (default: 256)")
    return parser.parse_args()


# Run training
if __name__ == "__main__":
    args = parse_args()
    Config.from_args(args)

    print(f"\n✅ Configuration loaded!")
    print(f"   Device: {Config.DEVICE}")
    print(f"   Model: {Config.MODEL_NAME} with {Config.ENCODER} encoder")
    print(f"   Image size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print(f"   Data: {Config.DATA_DIR}")
    print(f"   Output: {Config.OUTPUT_DIR}")

    Config.setup_dirs()
    model, best_iou = main()
