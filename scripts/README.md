# LingBot-Depth Scripts

## Overview

This folder contains real-time inference scripts for the LingBot-Depth project, designed to obtain depth estimation results from RGB-D cameras or monocular images.

> **Note**: The following scripts include personal modifications by me (mrblue), primarily focused on adjustments and optimizations for the real-time display interface.

---

## Script Descriptions

### realtime_depth.py

**Purpose**: Real-time depth display for Intel RealSense D435 camera

**Features**:
- RGBD Completion mode (RGB + sparse depth -> completed depth)
- RGB2Depth mode (monocular depth estimation from RGB only)
- 2x2 grid real-time display layout:
  - Top-left: RGB input image
  - Top-right: Model-refined depth map
  - Bottom-left: Raw depth from camera
  - Bottom-right: Depth difference map (refined - raw)
- Keyboard shortcuts: 1/2 to switch mode, s to save screenshot, q to quit

**Usage**:
```bash
python scripts/realtime_depth.py
python scripts/realtime_depth.py --width 848 --height 480
python scripts/realtime_depth.py --model robbyant/lingbot-depth-postrain-dc-vitl14
```

---

### rgb2depth.py

**Purpose**: Monocular depth estimation script that predicts depth from RGB images (without depth camera input)

**Usage**:
```bash
python scripts/rgb2depth.py --rgb path/to/rgb.png --intrinsics path/to/intrinsics.txt --output results
```

---

### rgbd_completion.py

**Purpose**: RGB-D depth completion script that takes RGB image and sparse depth map as input and outputs a completed depth map

**Usage**:
```bash
python scripts/rgbd_completion.py --rgb path/to/rgb.png --depth path/to/depth.png --intrinsics path/to/intrinsics.txt --output results
```

---

## Keyboard Shortcuts (realtime_depth.py)

| Key | Function |
|-----|----------|
| `1` | Switch to RGBD Completion mode |
| `2` | Switch to RGB2Depth monocular mode |
| `s` | Save current frame screenshot |
| `q` | Exit program |

---

## Personal Modifications

- **realtime_depth.py**:
  - Added 2x2 grid display layout
  - Added raw depth map display
  - Added depth difference visualization (refined - raw)
  - Optimized info bar display (separated FPS/mode and shortcuts)
  - Enhanced screenshot saving functionality

---

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- OpenCV
- pyrealsense2 (required only for realtime_depth.py)
- LingBot-Depth model

### pyrealsense2 Installation

For Ubuntu 24.04, follow this guide to install pyrealsense2:
[Ubuntu 24 安装 pyrealsense2](https://www.cnblogs.com/adrow/p/18319909)

---

## Recommended Configuration

### NVIDIA RTX 40 Series GPUs

For optimal real-time performance with LingBot-Depth, the following RTX 40 series GPUs are recommended:

| GPU | VRAM | Performance | Notes |
|-----|------|-------------|-------|
| **RTX 4090** | 24GB | Excellent | Best performance, handles highest resolutions |
| **RTX 4080** | 16GB | Very Good | Excellent balance of price/performance |
| **RTX 4070 Ti** | 12GB | Good | Good for 640x480 @ 30fps |
| **RTX 4070** | 12GB | Good | Minimum recommended for real-time |
| **RTX 4060 Ti** | 8GB | Moderate | May need reduced resolution |

### Recommended Settings for RTX 40 Series

```bash
# For RTX 4090/4080 (high performance)
python scripts/realtime_depth.py --width 848 --height 480 --fps 30

# For RTX 4070 series (balanced)
python scripts/realtime_depth.py --width 640 --height 480 --fps 30
```

### Performance Tips

- Use CUDA (default) for GPU acceleration
- bfloat16 mixed precision is enabled by default for better throughput
- Lower resolution if FPS drops below target

---

## Known Issues

### RTX 5090D Compatibility

**Important**: Due to library conflicts, sm_120 architecture is not currently supported.
The following error may occur on RTX 5090D:

```
CUDA error: no kernel image is available for execution on the device
```

This is caused by PyTorch/TorchVision not having pre-built wheels for the new sm_120 architecture.

**Workaround**:
- Use RTX 40 series (sm_89) which is fully supported
- RTX 5090D support will be added once PyTorch releases compatible builds

---

*README created by mrblue | Last updated: February 2026*
