# Configuration settings for 3D pose estimation pipeline

import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Joint definitions
LEFT_JOINTS = {1, 2, 3, 11, 12, 13}
RIGHT_JOINTS = {4, 5, 6, 14, 15, 16}

# COCO skeleton connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Video processing settings
VIDEO_CONFIG = {
    'input_size': (640, 360),
    'output_3d_size': 720,
    'fps_range': (5, 120),
    'default_fps': 30,
    'conf_threshold': 0.25,
}

# Smoothing parameters
SMOOTHING_CONFIG = {
    '2d_window_length': 11,
    '2d_polyorder': 3,
    '3d_window_length': 11,
    '3d_polyorder': 3,
}

# 3D lifting parameters
LIFTING_CONFIG = {
    'chunk_size': 243,
    'overlap': 60,
    'scale_factor': 1.5,
    'blend_factor': 0.5,
}

# Model paths
MODEL_CONFIG = {
    'yolo_model': "yolo11n-pose.pt",
    'motionbert_checkpoint': 'motionbert_ft_h36m.pth',
}

# Rendering settings
RENDER_CONFIG = {
    'panel_size': 720,
    'elev': 20,
    'azim': -60,
    'span_multiplier': 1.5,
    'min_span': 1.5,
}
