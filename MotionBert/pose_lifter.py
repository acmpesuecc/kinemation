# 3D pose lifting module using MotionBERT

import numpy as np
import torch

from config import DEVICE, LIFTING_CONFIG, MODEL_CONFIG
from pose_utils import create_anatomical_constraints, smooth_3d_poses, temporal_blend


def load_motionbert():
    # Load MotionBERT model for 3D pose lifting
    from lib.model.DSTformer import DSTformer
    
    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=2, num_joints=17, maxlen=243
    )
    
    checkpoint = torch.load(MODEL_CONFIG['motionbert_checkpoint'], map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    if DEVICE == 'cuda':
        model = model.half()
    
    return model


def postprocess_pose(pred_np):
    # Post-process 3D pose predictions with anatomical constraints
    out = pred_np.copy()

    out -= out[:, 0:1, :]
    
    out[:, :, 1] *= -1

    out = create_anatomical_constraints(out, LIFTING_CONFIG['blend_factor'])
    
    out *= LIFTING_CONFIG['scale_factor']
    
    return out


def lift_to_3d(kps_2d, model):
    # Lift 2D keypoints to 3D poses using MotionBERT
    N = len(kps_2d)
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    step = LIFTING_CONFIG['chunk_size'] - LIFTING_CONFIG['overlap']

    poses_acc = np.zeros((N, 17, 3), dtype=np.float32)
    count_acc = np.zeros(N, dtype=np.float32)

    for i, start in enumerate(range(0, N, step)):
        end = min(start + LIFTING_CONFIG['chunk_size'], N)
        chunk = kps_2d[start:end]
        T = len(chunk)

        if T < LIFTING_CONFIG['chunk_size']:
            pad = np.tile(chunk[-1:], (LIFTING_CONFIG['chunk_size'] - T, 1, 1))
            chunk = np.concatenate([chunk, pad], axis=0)

        tensor = torch.tensor(chunk[np.newaxis], dtype=dtype).to(DEVICE)

        with torch.no_grad():
            pred = model(tensor)

        pred_np = pred.squeeze(0).float().cpu().numpy()
        pred_np = postprocess_pose(pred_np)

        valid_len = end - start
        
        temporal_blend(
            poses_acc, count_acc, pred_np, start, end, 
            LIFTING_CONFIG['overlap']
        )

    poses_3d = poses_acc / np.maximum(count_acc, 1.0)[:, None, None]
    
    from pose_utils import smooth_3d_poses, SMOOTHING_CONFIG
    poses_3d = smooth_3d_poses(
        poses_3d, 
        SMOOTHING_CONFIG['3d_window_length'], 
        SMOOTHING_CONFIG['3d_polyorder']
    )
    
    np.save('poses_3d.npy', poses_3d)
    print(f"3D lifting done — shape: {poses_3d.shape}")
    return poses_3d
