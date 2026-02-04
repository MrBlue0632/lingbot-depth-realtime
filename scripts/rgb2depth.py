#!/usr/bin/env python3
"""
RGB2Depth Script

Uses LingBot-Depth for monocular depth estimation from RGB images only.
Generates dense depth prediction using RGB guidance without sparse depth input.

This script creates a virtual all-zero depth map as input, allowing the model
to predict depth from RGB alone using its learned prior.

Usage:
    python scripts/rgb2depth.py --rgb path/to/rgb.png --intrinsics path/to/intrinsics.txt --output output_dir

    python scripts/rgb2depth.py --rgb examples/0/rgb.png \
        --intrinsics examples/0/intrinsics.txt --output results_rgb2d
"""

import cv2
import torch
import numpy as np
import trimesh
import argparse
import time
from pathlib import Path
from mdm.model.v2 import MDMModel


def preprocess_rgb_image(image_path, device):
    """
    Load and preprocess RGB image.

    Args:
        image_path (str): Path to RGB image
        device (torch.device): Device to load tensor on

    Returns:
        tuple: (numpy_image, tensor_image)
            - numpy_image: RGB numpy array (H, W, 3), uint8
            - tensor_image: RGB tensor (1, 3, H, W), float32, [0,1]
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    # Read image and convert BGR to RGB
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Failed to read RGB image: {image_path}")

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.tensor(
        image_np / 255.0,
        dtype=torch.float32,
        device=device
    ).permute(2, 0, 1).unsqueeze(0)

    return image_np, image_tensor


def create_virtual_depth_map(height, width, device):
    """
    Create a virtual all-zero depth map as placeholder for RGB-only inference.

    The model expects depth input, so we provide zeros which will be treated
    as invalid regions and filled based on RGB guidance.

    Args:
        height (int): Image height
        width (int): Image width
        device (torch.device): Device to create tensor on

    Returns:
        torch.Tensor: Zero depth map (1, H, W), float32
    """
    depth = torch.zeros(1, height, width, dtype=torch.float32, device=device)
    return depth


def load_intrinsics(intrinsics_path, width, height):
    """
    Load camera intrinsics and normalize by image dimensions.

    Args:
        intrinsics_path (str): Path to intrinsics file (.txt or .json)
        width (int): Image width
        height (int): Image height

    Returns:
        np.ndarray: Normalized intrinsics matrix (3, 3)
    """
    if not Path(intrinsics_path).exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")

    # Load intrinsics
    if intrinsics_path.endswith('.json'):
        import json
        with open(intrinsics_path, 'r') as f:
            intrinsics = np.array(json.load(f), dtype=np.float32)
    else:
        intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)

    # Normalize by image dimensions
    intrinsics_normalized = intrinsics.copy()
    intrinsics_normalized[0, 0] /= width   # fx
    intrinsics_normalized[0, 2] /= width   # cx
    intrinsics_normalized[1, 1] /= height  # fy
    intrinsics_normalized[1, 2] /= height  # cy

    return intrinsics_normalized


def depth_to_color(depth_map, vmin=None, vmax=None):
    """
    Convert depth map to color visualization using OpenCV colormap.

    Args:
        depth_map (np.ndarray): Depth map (H, W)
        vmin (float): Minimum depth for colormap (auto if None)
        vmax (float): Maximum depth for colormap (auto if None)

    Returns:
        np.ndarray: Colored depth map (H, W, 3) in BGR format
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Auto-range if not specified
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap (TURBO provides good color distribution)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored


def main():
    """Main function with argument parsing and execution."""
    parser = argparse.ArgumentParser(
        description='RGB2Depth: Monocular Depth Estimation from RGB Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with example data
  python scripts/rgb2depth.py --rgb examples/0/rgb.png \
      --intrinsics examples/0/intrinsics.txt --output results_rgb2d

  # Use custom data
  python scripts/rgb2depth.py --rgb path/to/rgb.png \
      --intrinsics path/to/intrinsics.txt \
      --output my_results

  # Use Hugging Face model instead of local checkpoint
  python scripts/rgb2depth.py --rgb examples/0/rgb.png \
      --intrinsics examples/0/intrinsics.txt \
      --model robbyant/lingbot-depth-pretrain-vitl-14

Note:
  This script performs monocular depth estimation using only RGB input.
  The model generates dense depth predictions based on learned RGB-depth priors.
  For best results, use well-lit images with clear scene structure.
        """
    )

    parser.add_argument(
        '--rgb', type=str, required=True,
        help='Path to RGB image'
    )
    parser.add_argument(
        '--intrinsics', type=str, required=True,
        help='Path to camera intrinsics file'
    )
    parser.add_argument(
        '--output', type=str, default='result_rgb2d',
        help='Output directory (default: result_rgb2d)'
    )
    parser.add_argument(
        '--model', type=str,
        default='ckpt/lingbot-depth-pretrain-vitl-14',
        help='Model path or Hugging Face ID (default: ckpt/lingbot-depth-pretrain-vitl-14)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto - uses CUDA if available)'
    )
    parser.add_argument(
        '--no-mask', action='store_true',
        help='Disable masking of invalid regions'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("RGB2Depth: Monocular Depth Estimation from RGB".center(70))
    print("=" * 70)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n[Device] {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n[Model] {args.model}")

    try:
        # Load data
        print(f"\n[Loading Data]")
        print(f"   RGB:        {args.rgb}")
        print(f"   Intrinsics: {args.intrinsics}")

        image_np, image_tensor = preprocess_rgb_image(args.rgb, device)

        h, w = image_np.shape[:2]
        print(f"   Image size: {w}x{h}")

        # Create virtual depth map (all zeros)
        depth_tensor = create_virtual_depth_map(h, w, device)
        print(f"   Virtual depth: all zeros (RGB-only inference)")

        # Load intrinsics
        intrinsics = load_intrinsics(args.intrinsics, w, h)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0)

        # Load model
        print(f"\n[Loading Model]")
        start_time = time.time()

        # Handle local checkpoint path - need to point to model.pt file
        model_path = args.model
        if Path(model_path).is_dir():
            model_path = str(Path(model_path) / 'model.pt')
            print(f"   Local checkpoint directory detected, using: {model_path}")

        model = MDMModel.from_pretrained(model_path).to(device)
        load_time = time.time() - start_time
        print(f"   Model loaded in {load_time:.2f}s")

        # Run inference
        print(f"\n[Running Inference]")
        print(f"   Mode: RGB-only (monocular depth estimation)")
        start_time = time.time()
        with torch.no_grad():
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                apply_mask=not args.no_mask,
                intrinsics=intrinsics_tensor
            )
        inference_time = time.time() - start_time

        depth_pred = output['depth'].squeeze().cpu().numpy()
        points_pred = output['points'].squeeze().cpu().numpy()

        print(f"   Inference completed in {inference_time:.3f}s")
        print(f"   Predicted depth range: {depth_pred[depth_pred > 0].min():.2f} - {depth_pred.max():.2f} meters")

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n[Saving Results] {output_dir}/")

        # 1. Save depth map as numpy array
        np.save(output_dir / 'depth_predicted.npy', depth_pred)
        print(f"   Depth array saved (.npy)")

        # 2. Save depth visualization
        depth_colored = depth_to_color(depth_pred)
        cv2.imwrite(str(output_dir / 'depth_predicted.png'), depth_colored)
        print(f"   Depth visualization saved (.png)")

        # 3. Save RGB image for reference
        cv2.imwrite(str(output_dir / 'rgb.png'), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # 4. Save point cloud
        valid_mask = np.isfinite(points_pred).all(axis=-1) & (points_pred[..., 2] > 0)
        verts = points_pred[valid_mask]
        verts_color = image_np[valid_mask]

        # Downsample for reasonable file size
        downsample = 2
        verts = verts[::downsample]
        verts_color = verts_color[::downsample]

        point_cloud = trimesh.PointCloud(verts, verts_color)
        point_cloud.export(output_dir / 'point_cloud.ply')
        print(f"   Point cloud saved ({len(verts):,} points)")

        # Print summary
        print(f"\n[Summary]")
        print(f"   Model load time:   {load_time:.2f}s")
        print(f"   Inference time:    {inference_time:.3f}s")
        print(f"   Valid points:      {valid_mask.sum():,} / {valid_mask.size:,}")
        print(f"   Mode:              RGB-only (monocular)")

        print(f"\n[Done] Results saved to: {output_dir}/")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return 1
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
