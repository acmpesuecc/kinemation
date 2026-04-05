# Main entry point for 3D pose estimation pipeline

import warnings
warnings.filterwarnings('ignore')

from pose_detector import extract_keypoints
from pose_lifter import load_motionbert, lift_to_3d
from pose_renderer import render_3d_video


def main():
    # Main processing pipeline for 3D pose estimation
    print("Starting 3D Pose Estimation Pipeline")
    
    try:
        print("\ntep 1: Extracting 2D keypoints...")
        kps_2d, fps, frame_size = extract_keypoints('test.mp4')
        
        print("\nStep 2: Loading MotionBERT model...")
        motionbert = load_motionbert()
        
        print("\nStep 3: Lifting to 3D...")
        poses_3d = lift_to_3d(kps_2d, motionbert)
        
        print("\nStep 4: Rendering output video...")
        output_path = render_3d_video(
            'test.mp4', 
            poses_3d, 
            fps, 
            frame_size, 
            'output_3d.mp4'
        )
        
        print(f"\n Processing complete!")
        print(f" Final pose shape: {poses_3d.shape}")
        print(f"Output saved: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
