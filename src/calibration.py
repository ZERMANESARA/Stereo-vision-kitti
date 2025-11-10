"""
KITTI Stereo Calibration Parser

This module parses KITTI stereo dataset calibration files and extracts
camera parameters needed for stereo vision algorithms.

Key concepts:
- Projection matrix P: Maps 3D world points to 2D image points
- Intrinsic matrix K: Contains focal length and principal point
- Baseline B: Distance between left and right cameras
- Stereo setup: Two horizontally aligned cameras for depth estimation

Author: Sara Zermane
Date: November 2025
"""

import numpy as np
import os
from typing import Dict, Tuple


class KITTICalibration:
    """
    KITTI Calibration Reader
    
    Parses KITTI calibration files to extract stereo camera parameters.
    
    KITTI provides 4 cameras (0, 1, 2, 3):
    - Camera 2 (P2): Left color camera (main)
    - Camera 3 (P3): Right color camera (for stereo)
    
    The projection matrix P (3x4) has this structure:
        P = [fx  0   cx  tx]
            [0   fy  cy  ty]
            [0   0   1   0 ]
    
    Where:
    - fx, fy: Focal lengths in pixels (usually fx = fy)
    - cx, cy: Principal point (image center)
    - tx, ty: Translation (baseline for stereo)
    
    Attributes:
        calib_filepath (str): Path to calibration file
        P0, P1, P2, P3 (np.ndarray): Projection matrices (3x4)
        K_left, K_right (np.ndarray): Intrinsic matrices (3x3)
        focal_length (float): Focal length in pixels
        baseline (float): Stereo baseline in meters
        cx, cy (float): Principal point coordinates
    """
    
    def __init__(self, calib_filepath: str):
        """
        Initialize calibration from file.
        
        Args:
            calib_filepath: Path to KITTI calibration .txt file
            
        Raises:
            FileNotFoundError: If calibration file doesn't exist
            ValueError: If required matrices are missing
        """
        self.calib_filepath = calib_filepath
        
        # Initialize projection matrices (will be loaded from file)
        # P0, P1: Grayscale cameras (not used in this project)
        # P2: Left color camera (our main camera)
        # P3: Right color camera (for stereo matching)
        self.P0 = None
        self.P1 = None
        self.P2 = None
        self.P3 = None
        
        # Step 1: Load matrices from calibration file
        self._load_calibration()
        
        # Step 2: Extract useful parameters (focal, baseline, etc.)
        self._compute_parameters()
    
    def _load_calibration(self) -> None:
        """
        Load and parse calibration file.
        
        KITTI calibration file format (example line):
        P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ...
        
        This represents a 3x4 projection matrix stored as 12 values.
        """
        
        # Check if file exists
        if not os.path.exists(self.calib_filepath):
            raise FileNotFoundError(
                f"Calibration file not found: {self.calib_filepath}"
            )
        
        # Read all lines from file
        with open(self.calib_filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse each line to find projection matrices
        for line in lines:
            line = line.strip()  # Remove whitespace
            
            # Look for projection matrix lines
            # Format: "P_rect_XX: value1 value2 ... value12"
            if line.startswith('P_rect_00:'):
                self.P0 = self._parse_projection_matrix(line)
            elif line.startswith('P_rect_01:'):
                self.P1 = self._parse_projection_matrix(line)
            elif line.startswith('P_rect_02:'):
                # P2 = Left camera (main camera for stereo)
                self.P2 = self._parse_projection_matrix(line)
            elif line.startswith('P_rect_03:'):
                # P3 = Right camera (for stereo matching)
                self.P3 = self._parse_projection_matrix(line)
    
    def _parse_projection_matrix(self, line: str) -> np.ndarray:
        """
        Parse projection matrix from calibration file line.
        
        Input line format:
        "P_rect_02: 7.215e+02 0.000e+00 6.095e+02 4.485e+01 ..."
        
        Process:
        1. Split line by spaces
        2. Skip first element (label "P_rect_02:")
        3. Convert strings to floats
        4. Reshape 12 values into 3x4 matrix
        
        Args:
            line: Line containing projection matrix
            
        Returns:
            Projection matrix (3x4) as numpy array
        """
        # Split by whitespace: ["P_rect_02:", "7.215e+02", "0.000e+00", ...]
        values = line.split()[1:]  # Skip label, keep only numbers
        
        # Convert strings to floats: [721.5377, 0.0, 609.5593, ...]
        values = [float(v) for v in values]
        
        # Reshape 12 values into 3x4 matrix:
        # [val0  val1  val2  val3 ]
        # [val4  val5  val6  val7 ]
        # [val8  val9  val10 val11]
        P = np.array(values).reshape(3, 4)
        
        return P
    
    def _compute_parameters(self) -> None:
        """
        Compute derived camera parameters from projection matrices.
        
        From projection matrix P (3x4), we extract:
        1. Intrinsic matrix K (3x3): Internal camera parameters
        2. Focal length f: How much the camera "zooms"
        3. Principal point (cx, cy): Image center
        4. Baseline B: Distance between left and right cameras
        
        Mathematical background:
        ------------------------
        Projection matrix P can be decomposed as:
            P = K [R | t]
        
        Where:
        - K (3x3): Intrinsic matrix
        - R (3x3): Rotation matrix
        - t (3x1): Translation vector
        
        For KITTI rectified cameras, R = Identity, so:
            P = K [I | t] = [K | Kt]
        
        This means:
        - First 3 columns of P = K (intrinsic matrix)
        - Fourth column = K*t (translation)
        
        Stereo baseline:
        ---------------
        For horizontal stereo (cameras side-by-side):
        - Left camera P2: t = [0, 0, 0] (reference)
        - Right camera P3: t = [-B, 0, 0] (shifted by baseline)
        
        From P3[0,3] = fx * tx = fx * (-B), we get:
            B = -P3[0,3] / fx
        """
        
        # Check that we have the required matrices
        if self.P2 is None or self.P3 is None:
            raise ValueError("P2 and P3 matrices are required for stereo")
        
        # Extract intrinsic matrices K from first 3 columns of P
        # K is the same for both cameras in rectified stereo
        self.K_left = self.P2[:, :3]   # First 3 columns of P2
        self.K_right = self.P3[:, :3]  # First 3 columns of P3
        
        # Extract focal length from K[0,0]
        # Intrinsic matrix K has structure:
        #   [fx  0   cx]
        #   [0   fy  cy]
        #   [0   0   1 ]
        self.focal_length = self.P2[0, 0]  # fx (focal length in pixels)
        
        # Extract principal point (image center)
        self.cx = self.P2[0, 2]  # cx (horizontal center)
        self.cy = self.P2[1, 2]  # cy (vertical center)
        
        # Compute stereo baseline
        # P3[0,3] contains fx * (-baseline)
        # Therefore: baseline = -P3[0,3] / fx
        self.baseline = -self.P3[0, 3] / self.P3[0, 0]
        
        # Typical KITTI values:
        # - focal_length: ~720 pixels
        # - baseline: ~0.54 meters (54 cm)
        # - cx: ~600 pixels (for 1242x375 images)
        # - cy: ~180 pixels
    
    def get_stereo_params(self) -> Dict[str, any]:
        """
        Get essential stereo parameters for disparity-to-depth conversion.
        
        These parameters are needed for:
        - Stereo rectification
        - Disparity map computation
        - 3D reconstruction (disparity -> depth)
        
        Depth formula:
        -------------
        depth = (focal_length * baseline) / disparity
        
        Returns:
            Dictionary containing:
                - focal_length (float): Focal length in pixels
                - baseline (float): Stereo baseline in meters
                - cx, cy (float): Principal point coordinates
                - K_left, K_right (np.ndarray): Intrinsic matrices (3x3)
        """
        return {
            'focal_length': self.focal_length,  # f (pixels)
            'baseline': self.baseline,          # B (meters)
            'cx': self.cx,                      # Principal point x
            'cy': self.cy,                      # Principal point y
            'K_left': self.K_left,              # Left camera intrinsics
            'K_right': self.K_right             # Right camera intrinsics
        }
    
    def print_summary(self) -> None:
        """
        Print formatted calibration summary.
        
        Displays:
        - Camera parameters (focal, baseline, principal point)
        - Full projection matrices P2, P3
        - Intrinsic matrix K
        """
        
        print("=" * 70)
        print(f"KITTI Calibration: {os.path.basename(self.calib_filepath)}")
        print("=" * 70)
        
        # Camera parameters
        print("\nCamera Parameters:")
        print(f"  Focal length (f):  {self.focal_length:.2f} px")
        print(f"  Principal point:   cx={self.cx:.2f}, cy={self.cy:.2f}")
        print(f"  Baseline (B):      {self.baseline:.4f} m "
              f"({self.baseline*100:.2f} cm)")
        
        # Projection matrices
        print("\nProjection Matrices:")
        print("\nP2 (Left camera):")
        print(self.P2)
        print("\nP3 (Right camera):")
        print(self.P3)
        
        # Intrinsic matrix
        print("\nIntrinsic Matrix K:")
        print(self.K_left)
        print("=" * 70)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_scene(scene_id: str = '000000') -> KITTICalibration:
    """
    Test calibration parsing on a single scene.
    
    This function:
    1. Loads calibration file for specified scene
    2. Parses projection matrices
    3. Extracts camera parameters
    4. Displays results
    
    Args:
        scene_id: Scene identifier (e.g., '000000')
        
    Returns:
        KITTICalibration object with parsed parameters
    """
    print(f"\nTesting calibration for scene {scene_id}...\n")
    
    # Build path to calibration file
    calib_path = (
        f'data/data_scene_flow_calib/training/'
        f'calib_cam_to_cam/{scene_id}.txt'
    )
    
    # Parse calibration
    calib = KITTICalibration(calib_path)
    
    # Display summary
    calib.print_summary()
    
    return calib


def test_multiple_scenes(num_scenes: int = 5) -> Tuple[list, list]:
    """
    Test calibration parsing on multiple scenes.
    
    Purpose: Verify calibration consistency across scenes.
    KITTI cameras are fixed, so parameters should be identical
    across all scenes.
    
    Args:
        num_scenes: Number of scenes to test
        
    Returns:
        Tuple of (focal_lengths, baselines) lists for analysis
    """
    print(f"\nTesting {num_scenes} scenes...\n")
    
    # Lists to store parameters for statistics
    focal_lengths = []
    baselines = []
    
    # Loop through scenes
    for i in range(num_scenes):
        # Format scene ID with leading zeros: 000000, 000001, ...
        scene_id = f'{i:06d}'
        
        # Build calibration file path
        calib_path = (
            f'data/data_scene_flow_calib/training/'
            f'calib_cam_to_cam/{scene_id}.txt'
        )
        
        try:
            # Parse calibration
            calib = KITTICalibration(calib_path)
            
            # Store parameters
            focal_lengths.append(calib.focal_length)
            baselines.append(calib.baseline)
            
            # Display brief summary
            print(f"Scene {scene_id}: "
                  f"f={calib.focal_length:.2f}px, "
                  f"B={calib.baseline*100:.2f}cm")
        
        except Exception as e:
            print(f"Scene {scene_id}: Error - {e}")
    
    # Compute and display statistics
    print("\n" + "=" * 70)
    print("Statistics (should show very low variance):")
    print("=" * 70)
    print(f"Mean focal length:  {np.mean(focal_lengths):.2f} "
          f"+/- {np.std(focal_lengths):.2f} px")
    print(f"Mean baseline:      {np.mean(baselines)*100:.2f} "
          f"+/- {np.std(baselines)*100:.2f} cm")
    print("=" * 70)
    
    return focal_lengths, baselines


def save_calibration_summary(output_file: str = 'calibration_summary.txt') -> None:
    """
    Save calibration summary for all scenes to file.
    
    Creates a CSV-like text file with calibration parameters
    for all scenes in the dataset.
    
    Useful for:
    - Quick reference
    - Debugging
    - Verification of calibration consistency
    
    Args:
        output_file: Output filename
    """
    print(f"\nSaving calibration summary to {output_file}...\n")
    
    # Get list of all calibration files
    calib_dir = 'data/data_scene_flow_calib/training/calib_cam_to_cam'
    calib_files = sorted([
        f for f in os.listdir(calib_dir) 
        if f.endswith('.txt')
    ])
    
    # Open output file
    with open(output_file, 'w') as f:
        # Write header
        f.write("KITTI Calibration Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Scene':<10} {'Focal (px)':<15} {'Baseline (m)':<15} "
                f"{'cx':<10} {'cy':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Process each scene
        for calib_file in calib_files:
            scene_id = calib_file.replace('.txt', '')
            calib_path = os.path.join(calib_dir, calib_file)
            
            try:
                # Parse calibration
                calib = KITTICalibration(calib_path)
                
                # Write parameters to file
                f.write(
                    f"{scene_id:<10} "
                    f"{calib.focal_length:<15.2f} "
                    f"{calib.baseline:<15.4f} "
                    f"{calib.cx:<10.2f} "
                    f"{calib.cy:<10.2f}\n"
                )
            except Exception as e:
                # Log errors
                f.write(f"{scene_id:<10} ERROR: {e}\n")
    
    print(f"Summary saved to {output_file}")


def main():
    """
    Main function to run calibration tests.
    
    Runs three tests:
    1. Detailed analysis of one scene
    2. Consistency check across multiple scenes
    3. Optional export of all calibrations
    """
    
    print("\n" + "=" * 70)
    print("KITTI STEREO CALIBRATION PARSER")
    print("=" * 70)
    
    # Test 1: Single scene detailed analysis
    print("\n[TEST 1] Single scene analysis")
    print("Purpose: Verify parsing and display detailed parameters")
    calib = test_single_scene('000000')
    
    # Test 2: Multiple scenes consistency check
    print("\n\n[TEST 2] Multiple scenes consistency")
    print("Purpose: Verify calibration is consistent across scenes")
    focal_lengths, baselines = test_multiple_scenes(10)
    
    # Test 3: Optional full summary export
    print("\n\n[TEST 3] Export calibration summary")
    response = input("Export full calibration summary? (y/n): ")
    if response.lower() == 'y':
        save_calibration_summary()
    
    print("\n\nCalibration parsing completed successfully.")
    print("\nNext steps:")
    print("1. Use these parameters for stereo rectification")
    print("2. Compute disparity maps")
    print("3. Convert disparity to 3D points using: depth = (f*B)/disparity")


# Entry point
if __name__ == '__main__':
    main()