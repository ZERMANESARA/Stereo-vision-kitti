"""
KITTI Stereo Rectification

Rectifies stereo image pairs to align epipolar lines horizontally.
This simplifies stereo matching by reducing the search space from 2D to 1D.

Key concepts:
- Epipolar geometry: Geometric constraint between two views
- Rectification: Transformation that aligns epipolar lines with image rows
- After rectification, corresponding points lie on the same horizontal line

Author: Sara Zermane
Date: November 2025
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from calibration import KITTICalibration


class StereoRectifier:
    """
    Stereo image rectification using OpenCV.
    
    Rectification aligns stereo images so that:
    1. Epipolar lines are horizontal
    2. Corresponding points have the same y-coordinate
    3. Stereo matching becomes a 1D search problem
    
    Attributes:
        calib (KITTICalibration): Calibration object with camera parameters
        R1, R2 (np.ndarray): Rotation matrices for left and right images
        P1, P2 (np.ndarray): New projection matrices after rectification
        map1_left, map2_left: Rectification maps for left image
        map1_right, map2_right: Rectification maps for right image
    """
    
    def __init__(self, calib: KITTICalibration):
        """
        Initialize rectifier with calibration parameters.
        
        Args:
            calib: KITTICalibration object containing camera parameters
        """
        self.calib = calib
        
        # Rectification matrices (computed by stereoRectify)
        self.R1 = None  # Rotation for left camera
        self.R2 = None  # Rotation for right camera
        self.P1 = None  # New projection matrix for left
        self.P2 = None  # New projection matrix for right
        
        # Rectification maps (for fast remapping)
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
        
        # Compute rectification
        self._compute_rectification()
    
    def _compute_rectification(self):
        """
        Compute rectification parameters using OpenCV's stereoRectify.
        
        OpenCV's stereoRectify computes:
        - R1, R2: Rotation matrices to align cameras
        - P1, P2: New projection matrices for rectified images
        - Q: Disparity-to-depth mapping matrix (used later for 3D)
        
        The algorithm:
        1. Computes a rectification transformation
        2. Ensures epipolar lines are horizontal
        3. Minimizes image distortion
        """
        
        # Get camera parameters
        params = self.calib.get_stereo_params()
        K_left = params['K_left']
        K_right = params['K_right']
        
        # For KITTI rectified cameras:
        # - Rotation R = Identity (cameras already aligned)
        # - Translation T = [-baseline, 0, 0] (horizontal stereo)
        R = np.eye(3)  # No rotation between rectified cameras
        T = np.array([-params['baseline'], 0, 0])  # Translation = baseline
        
        # Compute image size from projection matrices
        # KITTI standard image size for rectified images
        image_size = (1242, 375)  # width, height
        
        # Compute rectification transformation
        # stereoRectify returns:
        # - R1, R2: Rotation matrices
        # - P1, P2: Projection matrices
        # - Q: Disparity-to-depth matrix
        # - roi1, roi2: Regions of interest (valid pixels)
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=K_left,
            distCoeffs1=None,  # KITTI images already undistorted
            cameraMatrix2=K_right,
            distCoeffs2=None,
            imageSize=image_size,
            R=R,
            T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,  # Maximize visible area
            alpha=0  # 0 = crop to valid pixels only
        )
        
        # Compute rectification maps for fast remapping
        # initUndistortRectifyMap creates lookup tables for remap()
        # This is faster than computing transformation for each pixel
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            cameraMatrix=K_left,
            distCoeffs=None,
            R=self.R1,
            newCameraMatrix=self.P1,
            size=image_size,
            m1type=cv2.CV_32FC1
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            cameraMatrix=K_right,
            distCoeffs=None,
            R=self.R2,
            newCameraMatrix=self.P2,
            size=image_size,
            m1type=cv2.CV_32FC1
        )
    
    def rectify_pair(self, img_left: np.ndarray, img_right: np.ndarray):
        """
        Rectify a stereo image pair.
        
        Process:
        1. Apply rectification maps to both images
        2. Result: Images with aligned horizontal epipolar lines
        
        Args:
            img_left: Left camera image (H x W x 3)
            img_right: Right camera image (H x W x 3)
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        
        # Apply rectification using precomputed maps
        # remap() is very fast (uses lookup tables)
        rect_left = cv2.remap(
            src=img_left,
            map1=self.map1_left,
            map2=self.map2_left,
            interpolation=cv2.INTER_LINEAR
        )
        
        rect_right = cv2.remap(
            src=img_right,
            map1=self.map1_right,
            map2=self.map2_right,
            interpolation=cv2.INTER_LINEAR
        )
        
        return rect_left, rect_right


def load_stereo_pair(scene_id: str, frame: str = '10'):
    """
    Load a stereo image pair from KITTI dataset.
    
    KITTI structure:
    - image_2/: Left camera (color)
    - image_3/: Right camera (color)
    - Format: XXXXXX_YY.png (scene_frame.png)
    
    Args:
        scene_id: Scene identifier (e.g., '000000')
        frame: Frame number (default '10')
        
    Returns:
        Tuple of (left_image, right_image) as numpy arrays (BGR format)
    """
    
    base_path = 'data/data_scene_flow/training'
    
    # Build file paths
    left_path = f'{base_path}/image_2/{scene_id}_{frame}.png'
    right_path = f'{base_path}/image_3/{scene_id}_{frame}.png'
    
    # Check if files exist
    if not os.path.exists(left_path):
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"Right image not found: {right_path}")
    
    # Load images (BGR format)
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    
    if img_left is None or img_right is None:
        raise ValueError("Failed to load images")
    
    return img_left, img_right


def visualize_rectification(img_left, img_right, rect_left, rect_right, 
                           save_path: str = None):
    """
    Visualize rectification results with epipolar lines.
    
    Draws horizontal lines to show that after rectification,
    corresponding points lie on the same horizontal line.
    
    Args:
        img_left: Original left image
        img_right: Original right image
        rect_left: Rectified left image
        rect_right: Rectified right image
        save_path: Optional path to save figure
    """
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Convert BGR to RGB for matplotlib
    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    rect_right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB)
    
    # Plot original images
    axes[0, 0].imshow(img_left_rgb)
    axes[0, 0].set_title('Original Left', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_right_rgb)
    axes[0, 1].set_title('Original Right', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot rectified images with epipolar lines
    axes[1, 0].imshow(rect_left_rgb)
    axes[1, 0].set_title('Rectified Left', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rect_right_rgb)
    axes[1, 1].set_title('Rectified Right', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Draw horizontal epipolar lines on rectified images
    height = rect_left.shape[0]
    line_positions = np.linspace(0, height, 10, dtype=int)
    
    for y in line_positions:
        # Left image lines (green)
        axes[1, 0].axhline(y=y, color='lime', linewidth=1, alpha=0.6)
        # Right image lines (green)
        axes[1, 1].axhline(y=y, color='lime', linewidth=1, alpha=0.6)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def save_rectified_pair(rect_left, rect_right, scene_id: str, 
                       output_dir: str = 'results/rectified'):
    """
    Save rectified image pair to disk.
    
    Args:
        rect_left: Rectified left image
        rect_right: Rectified right image
        scene_id: Scene identifier
        output_dir: Output directory
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    left_path = f'{output_dir}/{scene_id}_left_rect.png'
    right_path = f'{output_dir}/{scene_id}_right_rect.png'
    
    cv2.imwrite(left_path, rect_left)
    cv2.imwrite(right_path, rect_right)
    
    print(f"Saved rectified pair:")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")


def test_single_scene(scene_id: str = '000000'):
    """
    Test rectification on a single scene.
    
    Complete pipeline:
    1. Load calibration
    2. Load stereo pair
    3. Rectify images
    4. Visualize results
    5. Save rectified images
    
    Args:
        scene_id: Scene identifier (e.g., '000000')
    """
    
    print(f"\nRectifying scene {scene_id}...")
    print("=" * 60)
    
    # Step 1: Load calibration
    print("\n[1/5] Loading calibration...")
    calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
    calib = KITTICalibration(calib_path)
    print(f"  Focal: {calib.focal_length:.2f}px, Baseline: {calib.baseline*100:.2f}cm")
    
    # Step 2: Load stereo pair
    print("\n[2/5] Loading stereo pair...")
    img_left, img_right = load_stereo_pair(scene_id)
    print(f"  Image size: {img_left.shape[1]}x{img_left.shape[0]}")
    
    # Step 3: Initialize rectifier
    print("\n[3/5] Computing rectification...")
    rectifier = StereoRectifier(calib)
    print("  Rectification matrices computed")
    
    # Step 4: Rectify images
    print("\n[4/5] Rectifying images...")
    rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
    print("  Rectification complete")
    
    # Step 5: Save and visualize
    print("\n[5/5] Saving results...")
    save_rectified_pair(rect_left, rect_right, scene_id)
    
    print("\nVisualizing...")
    visualize_rectification(
        img_left, img_right, 
        rect_left, rect_right,
        save_path=f'results/rectified/rectification_comparison_{scene_id}.png'
    )
    
    print("\n" + "=" * 60)
    print("Rectification completed successfully!")
    print("\nNext step: Compute disparity maps")


def test_multiple_scenes(num_scenes: int = 3):
    """
    Test rectification on multiple scenes.
    
    Args:
        num_scenes: Number of scenes to process
    """
    
    print(f"\nProcessing {num_scenes} scenes...")
    print("=" * 60)
    
    for i in range(num_scenes):
        scene_id = f'{i:06d}'
        
        try:
            print(f"\nScene {scene_id}:")
            
            # Load and rectify
            calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
            calib = KITTICalibration(calib_path)
            
            img_left, img_right = load_stereo_pair(scene_id)
            
            rectifier = StereoRectifier(calib)
            rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
            
            save_rectified_pair(rect_left, rect_right, scene_id)
            
            print(f"  Scene {scene_id} completed")
            
        except Exception as e:
            print(f"  Error processing scene {scene_id}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Processed {num_scenes} scenes")


def main():
    """Main function to run rectification tests."""
    
    print("\n" + "=" * 60)
    print("KITTI STEREO RECTIFICATION")
    print("=" * 60)
    
    # Test 1: Single scene with visualization
    print("\n[TEST 1] Single scene rectification")
    test_single_scene('000000')
    
    # Test 2: Multiple scenes (optional)
    print("\n\n[TEST 2] Multiple scenes")
    response = input("Process additional scenes? (y/n): ")
    if response.lower() == 'y':
        num = input("How many scenes? (1-10): ")
        try:
            test_multiple_scenes(int(num))
        except ValueError:
            print("Invalid number, skipping")
    
    print("\n\nRectification pipeline completed!")
    print("\nYou can now:")
    print("1. Check results/ folder for rectified images")
    print("2. Compare original vs rectified")
    print("3. Notice epipolar lines are now horizontal")


if __name__ == '__main__':
    main()