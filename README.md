# Stereo Vision Pipeline on KITTI Dataset

Complete implementation of a stereo vision pipeline for 3D reconstruction from stereo images.

## Overview

This project implements a full stereo vision pipeline using the KITTI Vision Benchmark dataset, including camera calibration parsing, stereo rectification, disparity map computation, and 3D point cloud reconstruction.

## Features

- **Calibration Parser**: Extract camera parameters from KITTI calibration files
- **Stereo Rectification**: Align stereo image pairs for simplified matching
- **Disparity Computation**: Calculate depth maps using stereo matching algorithms
- **3D Reconstruction**: Generate point clouds from disparity maps (WIP)

## Tech Stack

- Python 3.9+
- OpenCV 4.8+
- NumPy
- Matplotlib

## Project Structure
```
stereo-vision-kitti/
├── src/
│   ├── calibration.py      # Camera calibration parser
│   ├── rectification.py    # Stereo rectification
│   ├── disparity.py        # Disparity computation
│   └── reconstruction.py   # 3D reconstruction (WIP)
├── results/
│   ├── rectified/          # Rectified stereo pairs
│   └── disparity/          # Disparity maps
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone https://github.com/ZERMANESARA/stereo-vision-kitti.git
cd stereo-vision-kitti

# Create virtual environment
conda create -n stereo-vision python=3.9
conda activate stereo-vision

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/).

**Note:** Dataset not included in repository due to size. Download separately:
- `data_scene_flow.zip` (~2GB)
- `data_scene_flow_calib.zip` (~14MB)

Extract to `data/` folder.

## Usage

### 1. Calibration Parsing
```bash
python src/calibration.py
```

Extracts camera parameters from KITTI calibration files.

**Output parameters:**
- Focal length: 721.54 px
- Baseline: 47.06 cm
- Principal point: (609.56, 172.85)

### 2. Stereo Rectification
```bash
python src/rectification.py
```

Rectifies stereo image pairs to align epipolar lines horizontally.

**Result:**

![Rectification Example](results/rectified/rectification_comparison_000000.png)

*Before (top): Original stereo pair | After (bottom): Rectified with horizontal epipolar lines*

### 3. Disparity Computation
```bash
python src/disparity.py
```

Computes disparity maps using Semi-Global Block Matching (SGBM) algorithm.

**Result:**

![Disparity Example](results/disparity/disparity_visualization_000000.png)

*Left: Original image | Center: Grayscale disparity | Right: Color-coded disparity*

**Interpretation:**
- Warm colors (yellow/orange): Close objects (1-5m)
- Cool colors (red): Medium distance (5-20m)
- Dark/black: Far objects (>20m) or no correspondence

**Disparity to depth conversion:**
```
depth (meters) = (focal_length × baseline) / disparity
depth = (721.54 px × 0.4706 m) / disparity
```

### Algorithm Comparison

We tested both StereoBM and StereoSGBM algorithms:

![Algorithm Comparison](results/disparity/algorithm_comparison_000000.png)

**StereoBM (Fast):**
- Very fast computation (~2-3 seconds)
- Many artifacts in textureless regions
- Best for: Quick previews, real-time applications

**StereoSGBM (Accurate):**
- High-quality disparity maps
- Better handling of textureless regions
- Smooth depth transitions
- Best for: Production quality, offline processing

**Conclusion:** We use StereoSGBM for final results due to superior quality.

## Results

### Calibration
Successfully parses KITTI calibration files and extracts projection matrices, intrinsic parameters, focal length, and baseline.

### Rectification
Aligns stereo images so that corresponding points lie on the same horizontal line, simplifying stereo matching from a 2D to a 1D search problem.

### Disparity Computation
Generates depth maps with disparity range typically 1-255 pixels, corresponding to depth range of approximately 1-340 meters for KITTI dataset.

## Roadmap

- [x] Calibration parsing
- [x] Stereo rectification
- [x] Disparity map computation
- [ ] 3D point cloud reconstruction
- [ ] Interactive 3D visualization

## Author

**Sara Zermane**

Computer Vision Engineer | Multi-Camera Systems Specialist

[LinkedIn](https://www.linkedin.com/in/sara-zermane) | [GitHub](https://github.com/ZERMANESARA)

## License

This project is available for portfolio and educational purposes.

---

*Project developed as part of computer vision portfolio - November 2025*