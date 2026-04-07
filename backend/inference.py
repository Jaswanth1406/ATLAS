"""
ATLAS Road Segmentation Inference Engine
Production-ready inference wrapper for the trained UNet model.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import json
import os
import time


class RoadSegmentationInference:
    """Production inference class for road segmentation."""

    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.img_size = tuple(self.config["input_shape"])  # (256, 256)

        # Load TorchScript model
        if model_path.endswith(".pt"):
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            raise ValueError("Only TorchScript (.pt) models are supported in production.")

        self.model.to(self.device)
        self.model.eval()
        print(f"✅ ATLAS model loaded on {self.device}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess BGR/RGB image for inference."""
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image: np.ndarray, threshold: float = 0.5):
        """
        Run inference on a single image (RGB numpy array).

        Returns:
            mask: binary mask (H, W) uint8, 0 or 255
            prob_map: probability map (H, W) float32, 0-1
            inference_time_ms: time in milliseconds
        """
        original_h, original_w = image.shape[:2]

        start = time.perf_counter()

        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.model(input_tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Probability map
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        prob_map = cv2.resize(prob_map, (original_w, original_h))

        # Binary mask
        mask = (prob_map > threshold).astype(np.uint8) * 255

        return mask, prob_map, elapsed_ms

    def create_overlay(
        self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
        color: tuple = (0, 255, 100)
    ) -> np.ndarray:
        """
        Create a colored overlay of the road mask on the original image.

        Args:
            image: RGB image (H, W, 3)
            mask: binary mask (H, W) with 0/255
            alpha: overlay transparency
            color: RGB color for road highlight

        Returns:
            overlay: RGB image (H, W, 3)
        """
        overlay = image.copy()
        road_region = mask > 0

        # Create colored mask
        color_mask = np.zeros_like(image)
        color_mask[road_region] = color

        # Blend
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

        # Add border glow on road edges
        edges = cv2.Canny(mask, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        overlay[edges_dilated > 0] = [0, 255, 200]

        return overlay
