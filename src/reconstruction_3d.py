"""
KITTI 3D Reconstruction

Reconstructs 3D point clouds from disparity maps using triangulation.
Converts 2D image coordinates + disparity into 3D world coordinates.

Key concepts:
- Triangulation: Computing 3D position from stereo correspondences
- Point cloud: Collection of 3D points (X, Y, Z) with color (R, G, B)
- Depth formula: Z = (focal_length × baseline) / disparity
- 3D projection: X = (x - cx) × Z / f, Y = (y - cy) × Z / f

Author: Sara Zermane
Date: November 2025
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calibration import KITTICalibration
from rectification import StereoRectifier, load_stereo_pair
from disparity import DisparityComputer


class PointCloudReconstructor:
    """
    Reconstructs 3D point clouds from disparity maps.
    
    Uses triangulation to convert 2D image coordinates + disparity
    into 3D world coordinates (X, Y, Z).
    
    Coordinate system:
    - X: Right (horizontal in image)
    - Y: Down (vertical in image)
    - Z: Forward (depth, away from camera)
    
    Attributes:
        calib: KITTICalibration object with camera parameters
        focal_length: Focal length in pixels
        baseline: Stereo baseline in meters
        cx, cy: Principal point coordinates
    """
    
    def __init__(self, calib: KITTICalibration):
        """
        Initialize reconstructor with calibration parameters.
        
        Args:
            calib: KITTICalibration object
        """
        self.calib = calib
        params = calib.get_stereo_params()
        
        self.focal_length = params['focal_length']
        self.baseline = params['baseline']
        self.cx = params['cx']
        self.cy = params['cy']
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Formula: Z = (f × B) / d
        Where:
        - Z: depth in meters
        - f: focal length in pixels
        - B: baseline in meters
        - d: disparity in pixels
        
        Args:
            disparity: Disparity map (float, pixels)
            
        Returns:
            Depth map (float, meters)
        """
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # Compute depth
        depth = (self.focal_length * self.baseline) / disparity_safe
        
        # Mark invalid depths
        depth[disparity <= 0] = 0
        
        return depth
    
    def reconstruct_point_cloud(self, disparity: np.ndarray, 
                                image: np.ndarray,
                                max_depth: float = 50.0) -> tuple:
        """
        Reconstruct 3D point cloud from disparity map.
        
        Process:
        1. Convert disparity to depth (Z)
        2. For each valid pixel (u, v):
           - Compute X = (u - cx) × Z / f
           - Compute Y = (v - cy) × Z / f
           - Store point (X, Y, Z) with color from image
        
        Args:
            disparity: Disparity map (float32, non-normalized)
            image: Color image (BGR) for point colors
            max_depth: Maximum depth to include (meters)
            
        Returns:
            Tuple of (points, colors)
            - points: (N, 3) array of XYZ coordinates
            - colors: (N, 3) array of RGB colors (0-1 range)
        """
        # Get image dimensions
        height, width = disparity.shape
        
        # Convert disparity to depth
        depth = self.disparity_to_depth(disparity)
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to float for calculations
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        
        # Compute 3D coordinates
        # X: horizontal (right positive)
        # Y: vertical (down positive)
        # Z: depth (forward positive)
        X = (u - self.cx) * depth / self.focal_length
        Y = (v - self.cy) * depth / self.focal_length
        Z = depth
        
        # Filter valid points
        # Valid if: disparity > 0 and depth < max_depth
        valid_mask = (disparity > 0) & (depth > 0) & (depth < max_depth)
        
        # Extract valid 3D points
        points = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)
        
        # Extract colors from image (convert BGR to RGB and normalize)
        if len(image.shape) == 3:
            colors = image[valid_mask] / 255.0
            colors = colors[:, ::-1]  # BGR to RGB
        else:
            # Grayscale: replicate to RGB
            gray = image[valid_mask] / 255.0
            colors = np.stack([gray, gray, gray], axis=1)
        
        return points, colors
    
    def filter_point_cloud(self, points: np.ndarray, colors: np.ndarray,
                          downsample: int = 1,
                          min_z: float = 1.0,
                          max_z: float = 50.0) -> tuple:
        """
        Filter and downsample point cloud.
        
        Filtering:
        - Remove points too close (min_z)
        - Remove points too far (max_z)
        - Downsample by factor (for performance)
        
        Args:
            points: (N, 3) array of XYZ coordinates
            colors: (N, 3) array of RGB colors
            downsample: Downsampling factor (1 = no downsample)
            min_z: Minimum depth (meters)
            max_z: Maximum depth (meters)
            
        Returns:
            Tuple of (filtered_points, filtered_colors)
        """
        # Filter by depth
        z_mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
        points = points[z_mask]
        colors = colors[z_mask]
        
        # Downsample
        if downsample > 1:
            indices = np.arange(0, len(points), downsample)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors


def visualize_point_cloud(points: np.ndarray, colors: np.ndarray,
                         save_path: str = None,
                         title: str = "3D Point Cloud"):
    """
    Visualize 3D point cloud using matplotlib.
    
    Creates interactive 3D plot with:
    - Colored points
    - Axis labels
    - Grid
    - Rotatable view
    
    Args:
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors
        save_path: Optional path to save figure
        title: Plot title
    """
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    # Downsample for display if too many points
    max_points = 50000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_display = points[indices]
        colors_display = colors[indices]
    else:
        points_display = points
        colors_display = colors
    
    ax.scatter(points_display[:, 0],  # X
              points_display[:, 1],   # Y
              points_display[:, 2],   # Z
              c=colors_display,
              s=1,  # Point size
              alpha=0.6)
    
    # Set labels
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([
        points_display[:, 0].max() - points_display[:, 0].min(),
        points_display[:, 1].max() - points_display[:, 1].min(),
        points_display[:, 2].max() - points_display[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points_display[:, 0].max() + points_display[:, 0].min()) * 0.5
    mid_y = (points_display[:, 1].max() + points_display[:, 1].min()) * 0.5
    mid_z = (points_display[:, 2].max() + points_display[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to {save_path}")
    
    plt.show()


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray,
                         filepath: str):
    """
    Save point cloud to PLY format.
    
    PLY (Polygon File Format) is standard for 3D data.
    Can be opened in:
    - MeshLab
    - CloudCompare
    - Blender
    
    Args:
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors (0-1 range)
        filepath: Output file path (.ply)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert colors to 0-255 range
    colors_255 = (colors * 255).astype(np.uint8)
    
    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                   f"{colors_255[i, 0]} {colors_255[i, 1]} {colors_255[i, 2]}\n")
    
    print(f"Point cloud saved to {filepath}")
    print(f"Number of points: {len(points):,}")
    print(f"Open with MeshLab, CloudCompare, or Blender")


def test_single_scene(scene_id: str = '000000'):
    """
    Test 3D reconstruction on a single scene.
    
    Complete pipeline:
    1. Load calibration
    2. Load stereo pair
    3. Rectify images
    4. Compute disparity
    5. Reconstruct 3D point cloud
    6. Visualize and save
    
    Args:
        scene_id: Scene identifier
    """
    print(f"\n3D reconstruction for scene {scene_id}...")
    print("=" * 70)
    
    # Step 1: Load calibration
    print("\n[1/6] Loading calibration...")
    calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
    calib = KITTICalibration(calib_path)
    print(f"  Focal: {calib.focal_length:.2f}px, Baseline: {calib.baseline*100:.2f}cm")
    
    # Step 2: Load stereo pair
    print("\n[2/6] Loading stereo pair...")
    img_left, img_right = load_stereo_pair(scene_id)
    print(f"  Image size: {img_left.shape[1]}x{img_left.shape[0]}")
    
    # Step 3: Rectify
    print("\n[3/6] Rectifying images...")
    rectifier = StereoRectifier(calib)
    rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
    print("  Rectification complete")
    
    # Step 4: Compute disparity
    print("\n[4/6] Computing disparity...")
    disparity_computer = DisparityComputer(calib, algorithm='sgbm')
    
    # Get non-normalized disparity (float32) for 3D reconstruction
    disparity = disparity_computer.compute_disparity(rect_left, rect_right, normalize=False)
    print(f"  Disparity range: {disparity[disparity > 0].min():.1f} - {disparity.max():.1f} px")
    
    # Step 5: Reconstruct 3D
    print("\n[5/6] Reconstructing 3D point cloud...")
    reconstructor = PointCloudReconstructor(calib)
    points, colors = reconstructor.reconstruct_point_cloud(disparity, rect_left, max_depth=50.0)
    print(f"  Initial points: {len(points):,}")
    
    # Filter and downsample
    points, colors = reconstructor.filter_point_cloud(
        points, colors,
        downsample=5,  # Keep 1 in 5 points
        min_z=1.0,
        max_z=50.0
    )
    print(f"  Filtered points: {len(points):,}")
    
    # Step 6: Visualize and save
    print("\n[6/6] Saving results...")
    
    # Save PLY
    ply_path = f'results/3d/point_cloud_{scene_id}.ply'
    save_point_cloud_ply(points, colors, ply_path)
    
    # Visualize
    print("\nVisualizing 3D point cloud...")
    print("(Rotate with mouse, close window to continue)")
    visualize_point_cloud(
        points, colors,
        save_path=f'results/3d/point_cloud_visualization_{scene_id}.png',
        title=f'3D Point Cloud - Scene {scene_id}'
    )
    
    print("\n" + "=" * 70)
    print("3D reconstruction completed successfully!")
    print(f"\nGenerated files:")
    print(f"  PLY file: {ply_path}")
    print(f"  Visualization: results/3d/point_cloud_visualization_{scene_id}.png")


def create_multiple_views(scene_id: str = '000000'):
    """
    Create multiple views of the point cloud from different angles.
    
    Args:
        scene_id: Scene identifier
    """
    print(f"\nCreating multiple views for scene {scene_id}...")
    
    # Full pipeline (reuse from test_single_scene)
    calib_path = f'data/data_scene_flow_calib/training/calib_cam_to_cam/{scene_id}.txt'
    calib = KITTICalibration(calib_path)
    
    img_left, img_right = load_stereo_pair(scene_id)
    rectifier = StereoRectifier(calib)
    rect_left, rect_right = rectifier.rectify_pair(img_left, img_right)
    
    disparity_computer = DisparityComputer(calib, algorithm='sgbm')
    disparity = disparity_computer.compute_disparity(rect_left, rect_right, normalize=False)
    
    reconstructor = PointCloudReconstructor(calib)
    points, colors = reconstructor.reconstruct_point_cloud(disparity, rect_left)
    points, colors = reconstructor.filter_point_cloud(points, colors, downsample=5)
    
    # Create views from different angles
    angles = [
        (20, 45, "Front View"),
        (20, 135, "Side View"),
        (60, 45, "Top View"),
        (5, 0, "Driver View")
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    for idx, (elev, azim, title) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Downsample for display
        max_points = 20000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            pts = points[indices]
            cols = colors[indices]
        else:
            pts = points
            cols = colors
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1, alpha=0.6)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'results/3d/multiple_views_{scene_id}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Multiple views saved to {save_path}")
    plt.show()


def main():
    """Main function to run 3D reconstruction tests."""
    
    print("\n" + "=" * 70)
    print("KITTI 3D RECONSTRUCTION")
    print("=" * 70)
    
    # Test 1: Single scene reconstruction
    print("\n[TEST 1] Single scene 3D reconstruction")
    test_single_scene('000000')
    
    # Test 2: Multiple views (optional)
    print("\n\n[TEST 2] Multiple viewing angles")
    response = input("Generate multiple views? (y/n): ")
    if response.lower() == 'y':
        create_multiple_views('000000')
    
    print("\n\n3D reconstruction pipeline completed!")
    print("\nYou now have:")
    print("1. PLY point cloud file (open in MeshLab/CloudCompare/Blender)")
    print("2. 3D visualization images")
    print("3. Complete stereo vision pipeline!")


if __name__ == '__main__':
    main()