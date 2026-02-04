#!/usr/bin/env python3
"""
Real-time Depth Display for Intel RealSense D435

Supports two modes:
1. RGBD Completion: RGB + sparse depth -> completed depth
2. RGB2Depth: RGB only -> monocular depth estimation

Controls:
    1: Switch to RGBD Completion mode
    2: Switch to RGB2Depth mode (monocular)
    s: Save screenshot
    q: Quit

Usage:
    python scripts/realtime_depth.py
    python scripts/realtime_depth.py --width 848 --height 480
    python scripts/realtime_depth.py --model robbyant/lingbot-depth-postrain-dc-vitl14
"""

import cv2
import torch
import numpy as np
import argparse
import time
import threading
import signal
import sys
from pathlib import Path
from mdm.model.v2 import MDMModel

# RealSense library
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class D435Camera:
    """Intel RealSense D435 camera wrapper."""

    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize D435 camera.

        Args:
            width: Image width
            height: Image height
            fps: Frame rate
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.color_frame = None
        self.depth_frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start camera and frame acquisition thread."""
        if not REALSENSE_AVAILABLE:
            raise ImportError(
                "pyrealsense2 not installed. Install with: "
                "pip install pyrealsense2"
            )

        # Configure streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps
        )
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps
        )

        # Start streaming
        profile = self.pipeline.start(config)

        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[Camera] Depth scale: {self.depth_scale}")

        # Start frame acquisition thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Allow camera to warm up
        time.sleep(2)
        print(f"[Camera] Started at {self.width}x{self.height} @ {self.fps}fps")

    def _capture_loop(self):
        """Continuously capture frames in background thread."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                with self.lock:
                    self.color_frame = frames.get_color_frame()
                    self.depth_frame = frames.get_depth_frame()
            except Exception:
                continue

    def read(self):
        """
        Read latest frame.

        Returns:
            tuple: (color_image, depth_image) or (None, None) if no frame
                - color_image: RGB numpy array (H, W, 3), uint8
                - depth_image: Depth numpy array (H, W), float32 in meters
        """
        with self.lock:
            if self.color_frame is None or self.depth_frame is None:
                return None, None

            # Get color image
            color_image = np.asanyarray(self.color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Get depth image and convert to meters
            depth_raw = np.asanyarray(self.depth_frame.get_data())
            depth_image = depth_raw.astype(np.float32) * self.depth_scale

            return color_image, depth_image

    def stop(self):
        """Stop camera and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.pipeline is not None:
            self.pipeline.stop()
        print("[Camera] Stopped")


class DepthProcessor:
    """Depth processing wrapper for LingBot-Depth model."""

    def __init__(self, model_path, device):
        """
        Initialize depth processor.

        Args:
            model_path: Path to model or Hugging Face model ID
            device: torch device
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        self.mode = "RGBD"  # RGBD or RGB2Depth

    def load_model(self):
        """Load the LingBot-Depth model."""
        print(f"[Model] Loading: {self.model_path}")

        # Handle local checkpoint path
        if Path(self.model_path).is_dir():
            actual_path = str(Path(self.model_path) / 'model.pt')
            print(f"[Model] Local checkpoint: {actual_path}")
        else:
            actual_path = self.model_path

        self.model = MDMModel.from_pretrained(actual_path).to(self.device)
        self.model.eval()
        print(f"[Model] Loaded on {self.device}")

    def set_mode(self, mode):
        """
        Set processing mode.

        Args:
            mode: "RGBD" or "RGB2Depth"
        """
        if mode != self.mode:
            self.mode = mode
            print(f"[Mode] Switched to {mode}")

    def process(self, rgb_image, depth_image=None):
        """
        Process image and return depth prediction.

        Args:
            rgb_image: RGB numpy array (H, W, 3), uint8
            depth_image: Depth numpy array (H, W), float32 in meters, or None

        Returns:
            dict: Contains 'depth' (H, W) and optionally 'points' (H, W, 3)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        h, w = rgb_image.shape[:2]

        # Prepare RGB tensor
        rgb_tensor = torch.tensor(
            rgb_image / 255.0,
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1).unsqueeze(0)

        # Prepare depth tensor
        if self.mode == "RGBD" and depth_image is not None:
            depth_tensor = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        else:
            # RGB2Depth mode: create virtual zero depth
            depth_tensor = torch.zeros(1, h, w, dtype=torch.float32, device=self.device)

        # Run inference
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=True
        ):
            output = self.model.infer(
                rgb_tensor,
                depth_in=depth_tensor,
                apply_mask=True
            )

        # Extract results
        depth_pred = output['depth'].squeeze().cpu().numpy()

        return {'depth': depth_pred}


def depth_to_color(depth_map, vmin=0.5, vmax=5.0):
    """
    Convert depth map to color visualization.

    Args:
        depth_map: Depth numpy array (H, W)
        vmin: Minimum depth for colormap
        vmax: Maximum depth for colormap

    Returns:
        numpy array: Colored image (H, W, 3) in BGR format
    """
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.1)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Normalize
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = [0, 0, 0]

    return colored


def depth_to_color_raw(depth_map, vmin=0.5, vmax=5.0):
    """
    Convert raw depth map to color visualization for display.

    Args:
        depth_map: Depth numpy array (H, W)
        vmin: Minimum depth for colormap
        vmax: Maximum depth for colormap

    Returns:
        numpy array: Colored image (H, W, 3) in BGR format
    """
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.1)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Normalize
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = [0, 0, 0]

    return colored


def compute_depth_difference(depth_refined, depth_raw, vmin=-1.0, vmax=1.0, diff_threshold=0.05):
    """
    Compute difference between refined and raw depth, return colored visualization.

    Features:
    - NaN/invalid areas in raw depth are marked in RED (indicates regions filled by model)
    - Only show difference colors when |diff| > threshold
    - Zero difference regions are shown in gray/black

    Args:
        depth_refined: Refined depth numpy array (H, W)
        depth_raw: Raw depth numpy array (H, W)
        vmin: Minimum difference value for colormap
        vmax: Maximum difference value for colormap
        diff_threshold: Minimum absolute difference to show color (default: 0.05m = 5cm)

    Returns:
        numpy array: Colored difference map (H, W, 3) in BGR format
        numpy array: Valid mask (H, W) boolean
    """
    # Create masks for different regions
    raw_nan_mask = ~np.isfinite(depth_raw) | (depth_raw <= 0.1)  # NaN or invalid in raw
    refined_valid = np.isfinite(depth_refined) & (depth_refined > 0.1)  # Valid in refined

    # Areas that were NaN in raw but are now valid in refined (filled by model)
    filled_mask = raw_nan_mask & refined_valid

    # Areas where both are valid
    both_valid = ~raw_nan_mask & refined_valid

    # Compute difference only where both are valid
    diff = np.zeros_like(depth_refined)
    diff[both_valid] = depth_refined[both_valid] - depth_raw[both_valid]

    # Create output image
    diff_colored = np.zeros((depth_refined.shape[0], depth_refined.shape[1], 3), dtype=np.uint8)

    # Step 1: Mark NaN/filled regions in RED
    diff_colored[filled_mask] = [0, 0, 255]  # BGR: Red

    # Step 2: Show difference colors where |diff| > threshold
    significant_diff = both_valid & (np.abs(diff) > diff_threshold)

    # Normalize significant differences for visualization
    diff_significant = diff.copy()
    diff_significant[~significant_diff] = 0

    diff_normalized = np.clip(
        (diff_significant - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap only to significant differences
    diff_map = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_TURBO)
    diff_map[~significant_diff] = 0

    # Blend: use diff_map where significant, red where filled
    diff_colored = np.where(
        significant_diff[:, :, np.newaxis] | filled_mask[:, :, np.newaxis],
        np.where(
            filled_mask[:, :, np.newaxis],
            np.array([0, 0, 255], dtype=np.uint8),  # Red for filled
            diff_map  # Color for significant diff
        ),
        diff_colored
    )

    # Add text label
    cv2.putText(
        diff_colored, "Diff: Refined - Raw (Red: Filled)", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    return diff_colored, both_valid | filled_mask


def create_2x2_display(rgb_image, depth_refined_colored, depth_raw_colored, depth_diff_colored, fps, mode):
    """
    Create 2x2 grid display for real-time depth visualization.

    Layout:
        [RGB Input]     [Refined Depth]
        [Raw Depth]     [Difference]

    Args:
        rgb_image: RGB input image (H, W, 3)
        depth_refined_colored: Refined depth visualization (H, W, 3) in BGR
        depth_raw_colored: Raw depth visualization (H, W, 3) in BGR
        depth_diff_colored: Difference visualization (H, W, 3) in BGR
        fps: Current FPS
        mode: Current processing mode

    Returns:
        numpy array: Combined display image
    """
    # Target size for each quadrant
    target_h = 240
    target_w = 320

    # Resize images to target size
    rgb_resized = cv2.resize(rgb_image, (target_w, target_h))
    depth_colored = cv2.resize(depth_refined_colored, (target_w, target_h))
    raw_colored = cv2.resize(depth_raw_colored, (target_w, target_h))
    diff_colored = cv2.resize(depth_diff_colored, (target_w, target_h))

    # Convert RGB to BGR for display
    rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

    # Create 2x2 grid
    top_row = np.concatenate([rgb_bgr, depth_colored], axis=1)
    bottom_row = np.concatenate([raw_colored, diff_colored], axis=1)
    display = np.concatenate([top_row, bottom_row], axis=0)

    # Add labels for each quadrant
    label_positions = [
        (5, 20, "RGB Input"),
        (target_w + 5, 20, "Refined Depth"),
        (5, target_h + 20, "Raw Depth"),
        (target_w + 5, target_h + 20, "Difference")
    ]

    for x, y, text in label_positions:
        cv2.putText(
            display, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # Add info bar at bottom
    info_h = 50
    info_bar = np.zeros((info_h, display.shape[1], 3), dtype=np.uint8)
    
    # Top line: FPS and Mode
    info_text = f"FPS: {fps:.1f} | Mode: {mode}"
    cv2.putText(
        info_bar, info_text, (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    
    # Bottom line: keyboard shortcuts
    legend_text = "[1] RGBD  [2] RGB2Depth  [s] Save  [q] Quit"
    cv2.putText(
        info_bar, legend_text, (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    display = np.concatenate([display, info_bar], axis=0)

    return display


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Depth Display for D435',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  1: RGBD Completion mode (RGB + sparse depth -> completed)
  2: RGB2Depth mode (RGB only -> monocular depth)
  s: Save screenshot
  q: Quit
        """
    )
    parser.add_argument(
        '--width', type=int, default=640,
        help='Image width (default: 640)'
    )
    parser.add_argument(
        '--height', type=int, default=480,
        help='Image height (default: 480)'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frame rate (default: 30)'
    )
    parser.add_argument(
        '--model', type=str,
        default='ckpt/lingbot-depth-pretrain-vitl-14',
        help='Model path or Hugging Face ID'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda)'
    )
    parser.add_argument(
        '--skip-model', action='store_true',
        help='Skip model loading (camera only mode)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Real-time Depth Display for Intel RealSense D435".center(60))
    print("=" * 60)

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"[Device] {device}")

    # Initialize camera
    print(f"[Camera] Initializing D435 at {args.width}x{args.height} @ {args.fps}fps")
    camera = D435Camera(width=args.width, height=args.height, fps=args.fps)

    # Initialize processor
    processor = None
    if not args.skip_model:
        processor = DepthProcessor(args.model, device)
        processor.load_model()
        processor.set_mode("RGBD")

    # Start camera
    try:
        camera.start()
    except Exception as e:
        print(f"[Error] Failed to start camera: {e}")
        print("[Tip] Make sure D435 is connected and pyrealsense2 is installed")
        return 1

    # Create display window
    window_name = "Real-time Depth Display - D435"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[Interrupted] Stopping...")
        camera.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    print("\n[Running] Press 'q' to quit, '1'/'2' to switch mode")
    fps_counter = []
    last_time = time.time()
    save_counter = 0

    try:
        while True:
            start_time = time.time()

            # Read frame
            rgb_image, depth_image = camera.read()
            if rgb_image is None:
                time.sleep(0.01)
                continue

            # Process if model loaded
            if processor is not None:
                output = processor.process(rgb_image, depth_image)
                depth_pred = output['depth']
            elif depth_image is not None:
                # Camera only mode: show raw depth
                depth_pred = depth_image
            else:
                depth_pred = np.zeros_like(rgb_image[:, :, 0], dtype=np.float32)

            # Convert depth to color visualization
            depth_colored = depth_to_color(depth_pred)

            # Compute depth difference (refined - raw)
            depth_diff, diff_mask = compute_depth_difference(depth_pred, depth_image)

            # Calculate FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - last_time))
            last_time = current_time
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)

            # Create raw depth visualization
            raw_colored = depth_to_color_raw(depth_image)

            # Create 2x2 grid display (pass pre-colored images)
            display = create_2x2_display(
                rgb_image, depth_colored, raw_colored, depth_diff, avg_fps,
                processor.mode if processor else "Camera"
            )

            # Show
            cv2.imshow(window_name, display)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1') and processor:
                processor.set_mode("RGBD")
            elif key == ord('2') and processor:
                processor.set_mode("RGB2Depth")
            elif key == ord('s'):
                # Save screenshot with 2x2 grid
                save_path = f"screenshot_{save_counter:04d}"
                Path(save_path).mkdir(exist_ok=True)
                
                # Save RGB
                cv2.imwrite(f"{save_path}/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                
                # Save depth data
                raw_depth = depth_image if depth_image is not None else np.zeros_like(depth_pred)
                np.save(f"{save_path}/depth_raw.npy", raw_depth)
                np.save(f"{save_path}/depth_refined.npy", depth_pred)
                
                # Save depth visualizations
                raw_colored = depth_to_color_raw(raw_depth)
                cv2.imwrite(f"{save_path}/depth_raw_colored.png", raw_colored)
                cv2.imwrite(f"{save_path}/depth_refined_colored.png", depth_colored)
                cv2.imwrite(f"{save_path}/depth_diff_colored.png", depth_diff)
                
                # Save 2x2 grid display
                cv2.imwrite(f"{save_path}/display_2x2.png", display)
                
                print(f"[Screenshot] Saved to {save_path}/")
                save_counter += 1

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[Done]")

    return 0


if __name__ == '__main__':
    exit(main())
