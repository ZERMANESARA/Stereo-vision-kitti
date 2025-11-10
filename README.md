\# Stereo Vision Pipeline on KITTI Dataset



Complete implementation of a stereo vision pipeline for 3D reconstruction from stereo images.



\## Overview



This project implements a full stereo vision pipeline using the KITTI Vision Benchmark dataset, including camera calibration parsing, stereo rectification, disparity map computation, and 3D point cloud reconstruction.



\## Features



\- \*\*Calibration Parser\*\*: Extract camera parameters from KITTI calibration files

\- \*\*Stereo Rectification\*\*: Align stereo image pairs for simplified matching

\- \*\*Disparity Computation\*\*: Calculate depth maps using stereo matching

\- \*\*3D Reconstruction\*\*: Generate point clouds from disparity maps



\## Tech Stack



\- Python 3.9+

\- OpenCV 4.8+

\- NumPy

\- Matplotlib



\## Project Structure

```

stereo-vision-kitti/

├── src/

│   ├── calibration.py      # Camera calibration parser

│   ├── rectification.py    # Stereo rectification (WIP)

│   ├── disparity.py        # Disparity computation (WIP)

│   └── reconstruction.py   # 3D reconstruction (WIP)

├── results/                # Output results

├── requirements.txt        # Python dependencies

└── README.md

```



\## Installation

```bash

\# Clone repository

git clone https://github.com/VOTRE\_USERNAME/stereo-vision-kitti.git

cd stereo-vision-kitti



\# Create virtual environment

conda create -n stereo-vision python=3.9

conda activate stereo-vision



\# Install dependencies

pip install -r requirements.txt

```



\## Usage



\### Calibration Parsing

```bash

python src/calibration.py

```



Extracts camera parameters (focal length, baseline, intrinsic matrices) from KITTI calibration files.



\*\*Output:\*\*

\- Focal length: 721.54 px

\- Baseline: 47.06 cm

\- Principal point: (609.56, 172.85)



\## Dataset



This project uses the \[KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/).



\*\*Note:\*\* Dataset not included in repository due to size. Download separately from KITTI website.



\## Roadmap



\- \[x] Calibration parsing

\- \[ ] Stereo rectification

\- \[ ] Disparity map computation

\- \[ ] 3D reconstruction

\- \[ ] Interactive visualization



\## Author



\*\*Sara Zermane\*\*



Computer Vision Engineer | Multi-Camera Systems Specialist



\[LinkedIn](https://www.linkedin.com/in/sara-zermane) | \[GitHub](https://github.com/SARAZERMANE)



\## License



This project is available for portfolio and educational purposes.



---



\*Project developed as part of computer vision portfolio - November 2025\*

