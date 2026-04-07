# 🔮 ATLAS — MODEL INFERENCE SCRIPT
# Test your trained model on images
#
# Usage:
#   python model_inference.py --model model.pt --config model_config.json --image road.jpg
#   python model_inference.py --model model.pt --config model_config.json --image_dir ./test_images/

import torch
import cv2
import numpy as np
from PIL import Image
import json
import os
import argparse
import glob


class RoadSegmentationInference:
    """Production inference class"""

    def __init__(self, model_path, config_path, device='cuda'):
        """
        Initialize inference

        Args:
            model_path: Path to model.pt or best_model.pth
            config_path: Path to model_config.json
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.img_size = tuple(self.config['input_shape'])

        # Load model
        if model_path.endswith('.pt'):
            # TorchScript
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            # PyTorch checkpoint
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(
                encoder_name=self.config['encoder'],
                encoder_weights=None,
                in_channels=3,
                classes=self.config['num_classes'],
                activation=None
            )

            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded on {self.device}")

    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize
        image = cv2.resize(image, self.img_size)

        # Normalize (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # To tensor [1, 3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image.to(self.device)

    def postprocess(self, output, threshold=0.5):
        """Postprocess model output to binary mask"""
        # Sigmoid + threshold
        mask = torch.sigmoid(output) > threshold
        mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        return mask

    def predict(self, image_path, threshold=0.5, return_prob=False):
        """
        Predict road mask for image

        Args:
            image_path: Path to image or numpy array
            threshold: Probability threshold (default: 0.5)
            return_prob: Return probability map instead of binary mask

        Returns:
            Binary mask (H, W) or probability map
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Could not read image: {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        original_size = (image.shape[1], image.shape[0])

        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        if return_prob:
            # Return probability map
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = cv2.resize(prob_map, original_size)
            return prob_map
        else:
            # Return binary mask
            mask = self.postprocess(output, threshold)
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            return mask

    def predict_batch(self, image_paths, threshold=0.5):
        """Predict on multiple images"""
        results = []
        for img_path in image_paths:
            mask = self.predict(img_path, threshold)
            results.append(mask)
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="ATLAS — Road Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python model_inference.py --model models/model.pt --config models/model_config.json --image test.jpg

  # Batch (directory)
  python model_inference.py --model models/model.pt --config models/model_config.json --image_dir ./test_images/

  # Custom threshold
  python model_inference.py --model models/model.pt --config models/model_config.json --image test.jpg --threshold 0.6

  # Save output to a specific directory
  python model_inference.py --model models/model.pt --config models/model_config.json --image test.jpg --output_dir ./results/
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pt or .pth)")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.json")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a directory of images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default="./inference_output", help="Directory to save outputs (default: ./inference_output)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.image and not args.image_dir:
        print("❌ Please provide --image <path> or --image_dir <path>")
        exit(1)

    # Initialize
    predictor = RoadSegmentationInference(args.model, args.config, device=args.device)

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.sort()

    if not image_paths:
        print("❌ No images found.")
        exit(1)

    print(f"\n📂 Processing {len(image_paths)} image(s)...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        mask = predictor.predict(img_path, threshold=args.threshold)
        if mask is None:
            continue

        # Save mask
        mask_path = os.path.join(args.output_dir, f"{basename}_mask.png")
        cv2.imwrite(mask_path, mask)

        # Create and save overlay
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        road_region = mask > 0
        color_mask = np.zeros_like(img_rgb)
        color_mask[road_region] = [0, 255, 100]
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.45, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = os.path.join(args.output_dir, f"{basename}_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)

        road_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        road_pct = road_pixels / total_pixels * 100

        print(f"   ✅ {os.path.basename(img_path)}: {road_pct:.1f}% road | mask → {mask_path}")

    print(f"\n✅ Done! Results saved to: {args.output_dir}")
