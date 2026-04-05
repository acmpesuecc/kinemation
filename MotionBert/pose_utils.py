# Utility functions for pose processing and coordinate transformations

import numpy as np
from scipy.signal import savgol_filter


def coco_to_h36m(kps):
    # Convert COCO keypoints to H36M format
    xy = kps[:, :2]
    conf = kps[:, 2:3]

    pelvis = (xy[11] + xy[12]) / 2
    pelvis_c = (conf[11] + conf[12]) / 2

    thorax = (xy[5] + xy[6]) / 2
    thorax_c = (conf[5] + conf[6]) / 2

    spine = (pelvis + thorax) / 2
    spine_c = (pelvis_c + thorax_c) / 2

    neck = (xy[5] + xy[6]) / 2 + np.array([0, -20])
    neck_c = (conf[5] + conf[6]) / 2

    head = xy[0] + np.array([0, -30])
    head_c = conf[0]

    h36m = np.array([
        pelvis,
        xy[12], xy[14], xy[16],
        xy[11], xy[13], xy[15],
        spine, thorax,
        neck, head,
        xy[5], xy[7], xy[9],
        xy[6], xy[8], xy[10],
    ])

    confs = np.array([
        pelvis_c,
        conf[12], conf[14], conf[16],
        conf[11], conf[13], conf[15],
        spine_c, thorax_c,
        neck_c, head_c,
        conf[5], conf[7], conf[9],
        conf[6], conf[8], conf[10],
    ])

    return np.concatenate([h36m, confs], axis=1)


def normalize_h36m(kps):
    # Normalize H36M keypoints by centering on pelvis and scaling
    xy = kps[:, :2].copy().astype(np.float32)
    conf = kps[:, 2:3].astype(np.float32)

    xy -= xy[0].copy()

    torso = np.linalg.norm(xy[8])
    if torso > 1e-6:
        xy /= torso
    
    return np.concatenate([xy, conf], axis=1)


def smooth_2d_keypoints(kps_2d, window_length=11, polyorder=3):
    # Apply Savitzky-Golay filter to smooth 2D keypoints
    smoothed = kps_2d.copy()
    
    for joint in range(17):
        for coord in range(2):
            if len(kps_2d) > window_length:
                smoothed[:, joint, coord] = savgol_filter(
                    kps_2d[:, joint, coord], 
                    window_length, 
                    polyorder
                )
    
    return smoothed


def smooth_3d_poses(poses_3d, window_length=11, polyorder=3):
    # Apply Savitzky-Golay filter to smooth 3D poses
    smoothed = poses_3d.copy()
    
    for joint in range(17):
        for coord in range(3):
            if len(poses_3d) > window_length:
                smoothed[:, joint, coord] = savgol_filter(
                    poses_3d[:, joint, coord], 
                    window_length, 
                    polyorder
                )
    
    return smoothed


def create_anatomical_constraints(out, blend_factor=0.5):
    # Apply anatomical constraints to ensure proper human pose structure
    for i in range(len(out)):
        pelvis = np.array([0.0, 0.0, 0.0])
        
        spine = np.array([0.0, 0.4, 0.0])
        chest = np.array([0.0, 0.6, 0.0])
        neck = np.array([0.0, 0.75, 0.0])
        head = np.array([0.0, 0.9, 0.0])
        
        lhip = np.array([0.15, 0.0, 0.0])
        rhip = np.array([-0.15, 0.0, 0.0])
        
        lknee = np.array([0.1, -0.4, 0.0])
        rknee = np.array([-0.1, -0.4, 0.0])
        lfoot = np.array([0.05, -0.8, 0.0])
        rfoot = np.array([-0.05, -0.8, 0.0])
        
        lshoulder = np.array([0.2, 0.65, 0.0])
        rshoulder = np.array([-0.2, 0.65, 0.0])
        
        lelbow = np.array([0.25, 0.3, 0.0])
        relbow = np.array([-0.25, 0.3, 0.0])
        lhand = np.array([0.3, 0.1, 0.0])
        rhand = np.array([-0.3, 0.1, 0.0])
        
        out[i, 0] = pelvis
        out[i, 1] = blend_factor * out[i, 1] + (1-blend_factor) * lhip
        out[i, 4] = blend_factor * out[i, 4] + (1-blend_factor) * rhip
        out[i, 2] = blend_factor * out[i, 2] + (1-blend_factor) * lknee
        out[i, 5] = blend_factor * out[i, 5] + (1-blend_factor) * rknee
        out[i, 3] = blend_factor * out[i, 3] + (1-blend_factor) * lfoot
        out[i, 6] = blend_factor * out[i, 6] + (1-blend_factor) * rfoot
        out[i, 8] = blend_factor * out[i, 8] + (1-blend_factor) * spine
        out[i, 7] = blend_factor * out[i, 7] + (1-blend_factor) * chest
        out[i, 9] = blend_factor * out[i, 9] + (1-blend_factor) * neck
        out[i, 10] = blend_factor * out[i, 10] + (1-blend_factor) * head
        out[i, 11] = blend_factor * out[i, 11] + (1-blend_factor) * lshoulder
        out[i, 14] = blend_factor * out[i, 14] + (1-blend_factor) * rshoulder
        out[i, 12] = blend_factor * out[i, 12] + (1-blend_factor) * lelbow
        out[i, 15] = blend_factor * out[i, 15] + (1-blend_factor) * relbow
        out[i, 13] = blend_factor * out[i, 13] + (1-blend_factor) * lhand
        out[i, 16] = blend_factor * out[i, 16] + (1-blend_factor) * rhand
    
    return out


def temporal_blend(poses_acc, count_acc, new_poses, start, end, overlap_size=60):
    # Blend overlapping regions for smooth transitions between chunks
    if start == 0 or end >= len(poses_acc):
        poses_acc[start:end] += new_poses[:end-start]
        count_acc[start:end] += 1.0
        return
    
    blend_start = max(start, start + overlap_size // 2)
    blend_end = min(end, end - overlap_size // 2)
    
    poses_acc[start:blend_start] += new_poses[:blend_start-start]
    count_acc[start:blend_start] += 1.0
    
    poses_acc[blend_end:end] += new_poses[blend_end-start:]
    count_acc[blend_end:end] += 1.0
    
    if blend_end > blend_start:
        blend_length = blend_end - blend_start
        for i in range(blend_length):
            alpha = (i + 1) / (blend_length + 1)
            poses_acc[blend_start + i] += (1 - alpha) * new_poses[blend_start - start + i]
            count_acc[blend_start + i] += (1 - alpha)
