"""
KITTI Disparity Computation

Computes disparity maps from rectified stereo image pairs using
block matching algorithms (StereoBM and StereoSGBM).

Key concepts:
- Disparity: Horizontal pixel difference between corresponding points
- Block Matching: Compares image patches to find correspondences
- Disparity map: Grayscale image where intensity = depth information
- Formula: depth = (focal_length × baseline) / disparity

Author: Sara Zermane
Date: November 2025
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from calibration import KITTICalibration
from rectification import StereoRectifier, load_stereo_pair


class DisparityComputer:
    """
    Computes disparity maps using OpenCV stereo matching algorithms.
    
    Two algorithms available:
    1. StereoBM: Fast, good for textured scenes
    2. StereoSGBM: Slower but more accurate, handles textureless regions
    
    Disparity represents the horizontal pixel shift between left and right images.
    Larger disparity = closer object, smaller disparity = farther object.
    
    Attributes:
        stereo_bm: StereoBM matcher (fast)
        stereo_sgbm: StereoSGBM matcher (accurate)
        focal_length: Camera focal length in pixels
        baseline: Stereo baseline in meters
    """
    
    def __init__(self, calib: KITTICalibration, algorithm: str = 'sgbm'):
        """
        Initialize disparity computer with calibration parameters.
        
        Args:
            calib: KITTICalibration object
            algorithm: 'bm' or 'sgbm' (default: 'sgbm')
        """
        self.calib = calib
        self.algorithm = algorithm
        
        params = calib.get_stereo_params()
        self.focal_length = params['focal_length']
        self.baseline = params['baseline']
        
        # Initialize matchers
        self._init_stereo_bm()
        self._init_stereo_sgbm()
    
    def _init_stereo_bm(self):
        """
        Initialize StereoBM (Block Matching) algorithm.
        
        StereoBM is fast but works best on textured images.
        Parameters:
        - numDisparities: Maximum disparity (must be divisible by 16)
        - blockSize: Size of matching block (odd number, typically 5-21)
        """
        self.stereo_bm = cv2.StereoBM_create(
            numDisparities=128,  # Max disparity search range
            blockSize=15         # Block size for matching
        )
        
        # Fine-tune parameters for better results
        self.stereo_bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
        self.stereo_bm.setPreFilterSize(9)
        self.stereo_bm.setPreFilterCap(31)
        self.stereo_bm.setMinDisparity(0)
        self.stereo_bm.setTextureThreshold(10)
        self.stereo_bm.setUniquenessRatio(15)
        self.stereo_bm.setSpeckleWindowSize(100)
        self.stereo_bm.setSpeckleRange(32)
        self.stereo_bm.setDisp12MaxDiff(1)
    
    def _init_stereo_sgbm(self):
        """
        Initialize StereoSGBM (Semi-Global Block Matching) algorithm.
        
        StereoSGBM is slower but more accurate than BM.
        Better at handling:
        - Textureless regions
        - Occlusions
        - Depth discontinuities
        
        Parameters:
        - minDisparity: Minimum disparity (usually 0)
        - numDisparities: Maximum disparity minus minimum (divisible by 16)
        - blockSize: Matched block size (odd, typically 3-11)
        - P1, P2: Smoothness parameters (penalize disparity changes)
        - uniquenessRatio: Margin for best match vs second best
        """
        
        # Parameters
        min_disp = 0
        num_disp = 128  # Must be divisible by 16
        block_size = 5  # Odd number, typically 3-11
        
        # Create SGBM matcher
        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,    # Smoothness parameter 1
            P2=32 * 3 * block_size ** 2,   # Smoothness parameter 2
            disp12MaxDiff=1,                # Max allowed difference (L-R check)
            uniquenessRatio=10,             # Margin in percentage
            speckleWindowSize=100,          # Max speckle size to filter
            speckleRange=32,                # Max disparity variation in speckle
            preFilterCap=63,                # Pre-filter cap
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3-way mode (more paths)
        )
    
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray,
                         normalize: bool = True) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair.
        
        Process:
        1. Convert images to grayscale (algorithms work on grayscale)
        2. Apply stereo matching algorithm
        3. Normalize disparity values to 0-255 range for visualization
        
        Args:
            img_left: Left rectified image (BGR or grayscale)
            img_right: Right rectified image (BGR or grayscale)
            normalize: If True, normalize disparity to 0-255 range
            
        Returns:
            Disparity map (float32 or uint8)
        """
        
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Compute disparity using selected algorithm
        if self.algorithm == 'bm':
            # StereoBM requires uint8 images
            disparity = self.stereo_bm.compute(gray_left, gray_right)
        else:
            # StereoSGBM works on uint8
            disparity = self.stereo_sgbm.compute(gray_left, gray_right)
        
        # Disparity is in fixed-point format (16-bit signed)
        # Convert to float and scale
        disparity = disparity.astype(np.float32) / 16.0
        
        # Normalize for visualization
        if normalize:
            # Remove invalid disparities (< 0)
            valid_mask = disparity > 0
            if valid_mask.any():
                min_disp = disparity[valid_mask].min()
                max_disp = disparity[valid_mask].max()
                
                # Normalize to 0-255
                disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)
                disparity_normalized[valid_mask] = (
                    255 * (disparity[valid_mask] - min_disp) / (max_disp - min_disp)
                ).astype(np.uint8)
                
                return disparity_normalized
            else:
                return np.zeros_like(disparity, dtype=np.uint8)
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map in meters.
        
        Formula: depth = (focal_length × baseline) / disparity
        
        Where:
        - focal_length: in pixels
        - baseline: in meters
        - disparity: in pixels
        - depth: in meters
        
        Args:
            disparity: Disparity map (non-normalized, float32)
            
        Returns:
            Depth map in meters (float32)
        """
        
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # Compute depth
        depth = (self.focal_length * self.baseline) / disparity_safe
        
        # Set invalid depths to infinity
        depth[disparity <= 0] = np.inf
        
        return depth


def visualize_disparity(img_left, disparity, save_path: str = None):
    """
    Visualize disparity map with color coding.
    
    Creates a figure showing:
    1. Original left image
    2. Disparity map (grayscale)
    3. Disparity map (color-coded for better visualization)
    
    Args:
        img_left: Original left image
        disparity: Disparity map (uint8, 0-255)
        save_path: Optional path to save figure
    """
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert BGR to RGB for matplotlib
    if len(img_left.shape) == 3:
        img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    else:
        img_left_rgb = img_left
    
    # 1. Original image
    axes[0].imshow(img_left_rgb, cmap='gray' if len(img_left.shape) == 2 else None)
    axes[0].set_title('Original Left Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Disparity grayscale
    axes[1].imshow(disparity, cmap='gray')
    axes[1].set_title('Disparity Map (Grayscale)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Disparity color-coded (hot colormap: white=close, black=far)
    im = axes[2].imshow(disparity, cmap='hot')
    axes[2].set_title('Disparity Map (Color-Coded)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Disparity (pixels)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def save_disparity_map(disparity, scene_id: str, 
                       output_dir: str = 'results/disparity'):
    """
    Save disparity map to disk.
    
    Args:
        disparity: Disparity map (uint8)
        scene_id: Scene identifier
        output_dir: Output directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save grayscale disparity
    gray_path = f'{output_dir}/{scene_id}_disparity_gray.png'
    cv2.imwrite(gray_path, disparity)
    
    # Save color-coded disparity
    disparity_color = cv2.applyColorMap(disparity, cv2.COLORMAP_HOT)
    color_path = f'{output_dir}/{scene_id}_disparity_color.png'
    cv2.imwrite(color_path, disparity_color)
    
    print(f"Saved disparity maps:")
    print(f"  Grayscale: {gray_path}")
    print(f"  Color:     {color_path}")


def test_single_scene(scene_id: str = '000000', algorithm: str = 'sgbm'):
    """
    Test disparity computation on a single scene.
    
    Complete pipeline:
    1. Load calibration
    2. Load stereo pair
    3. Rectify images
    4. Compute disparity
    5. Visualize and save results
    
    Args:
        scene_id: Scene identifier
        algorithm: 'bm' or 'sgbm'
    """
    
    print(f"\nComputing disparity for scene {scene_id} using {algorithm.upper()}...")
    print("=" * 70)
    
    # Step 1: Load calibration
    print("\n[1/5] Loading calibration...")
    calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
    calib = KITTICalibration(calib_path)
    print(f"  Focal: {calib.focal_length:.2f}px, Baseline: {calib.baseline*100:.2f}cm")
    
    # Step 2: Load stereo pair
    print("\n[2/5] Loading stereo pair...")
    img_left, img_right = load_stereo_pair(scene_id)
    print(f"  Image size: {img_left.shape[1]}x{img_left.shape[0]}")
    
    # Step 3: Rectify images
    print("\n[3/5] Rectifying images...")
    rectifier = StereoRectifier(calib)
    rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
    print("  Rectification complete")
    
    # Step 4: Compute disparity
    print("\n[4/5] Computing disparity...")
    disparity_computer = DisparityComputer(calib, algorithm=algorithm)
    disparity = disparity_computer.compute_disparity(rect_left, rect_right)
    print(f"  Disparity range: {disparity[disparity > 0].min():.1f} - {disparity.max():.1f} px")
    
    # Step 5: Save and visualize
    print("\n[5/5] Saving results...")
    save_disparity_map(disparity, scene_id)
    
    print("\nVisualizing...")
    visualize_disparity(
        rect_left, 
        disparity,
        save_path=f'results/disparity/disparity_visualization_{scene_id}.png'
    )
    
    print("\n" + "=" * 70)
    print("Disparity computation completed successfully!")
    print("\nNext step: 3D reconstruction from disparity")


def compare_algorithms(scene_id: str = '000000'):
    """
    Compare StereoBM vs StereoSGBM on same scene.
    
    Args:
        scene_id: Scene identifier
    """
    
    print(f"\nComparing BM vs SGBM on scene {scene_id}...")
    print("=" * 70)
    
    # Load and prepare
    calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
    calib = KITTICalibration(calib_path)
    
    img_left, img_right = load_stereo_pair(scene_id)
    rectifier = StereoRectifier(calib)
    rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
    
    # Compute with both algorithms
    print("\nComputing with StereoBM...")
    computer_bm = DisparityComputer(calib, algorithm='bm')
    disparity_bm = computer_bm.compute_disparity(rect_left, rect_right)
    
    print("Computing with StereoSGBM...")
    computer_sgbm = DisparityComputer(calib, algorithm='sgbm')
    disparity_sgbm = computer_sgbm.compute_disparity(rect_left, rect_right)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    img_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(img_left_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(disparity_bm, cmap='hot')
    axes[0, 1].set_title('StereoBM (Fast)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(disparity_sgbm, cmap='hot')
    axes[1, 0].set_title('StereoSGBM (Accurate)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference
    diff = cv2.absdiff(disparity_bm, disparity_sgbm)
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/disparity/algorithm_comparison_{scene_id}.png', dpi=150)
    plt.show()
    
    print("\nComparison saved!")


def main():
    """Main function to run disparity computation tests."""
    
    print("\n" + "=" * 70)
    print("KITTI DISPARITY COMPUTATION")
    print("=" * 70)
    
    # Test 1: Single scene with SGBM
    print("\n[TEST 1] Single scene disparity (SGBM)")
    test_single_scene('000000', algorithm='sgbm')
    
    # Test 2: Algorithm comparison (optional)
    print("\n\n[TEST 2] Algorithm comparison")
    response = input("Compare BM vs SGBM algorithms? (y/n): ")
    if response.lower() == 'y':
        compare_algorithms('000000')
    
    print("\n\nDisparity computation pipeline completed!")
    print("\nYou now have:")
    print("1. Grayscale disparity maps (depth information)")
    print("2. Color-coded disparity maps (visualization)")
    print("3. Ready for 3D reconstruction")


if __name__ == '__main__':
    main()