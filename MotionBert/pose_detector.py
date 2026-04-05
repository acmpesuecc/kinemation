# 2D pose detection and keypoint extraction module

import cv2
import numpy as np
from ultralytics import YOLO

from config import VIDEO_CONFIG
from pose_utils import coco_to_h36m, normalize_h36m, smooth_2d_keypoints


def draw_coco_skeleton(frame, person, conf_thresh=0.25):
    # Draw COCO skeleton on frame
    from config import COCO_CONNECTIONS
    
    for (a, b) in COCO_CONNECTIONS:
        xa, ya, ca = person[a]
        xb, yb, cb = person[b]
        if ca > conf_thresh and cb > conf_thresh:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)
    
    for (x, y, c) in person:
        if c > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


def extract_keypoints(video_path):
    # Extract 2D keypoints from video using YOLO pose detection
    yolo = YOLO(VIDEO_CONFIG['yolo_model'])
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps if VIDEO_CONFIG['fps_range'][0] < fps < VIDEO_CONFIG['fps_range'][1] else VIDEO_CONFIG['default_fps']
    W, H = VIDEO_CONFIG['input_size']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_2d = cv2.VideoWriter('output_2d.mp4', fourcc, fps, (W, H))

    all_kps = []
    last_valid_raw = None
    last_valid_norm = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))
        frame_idx += 1
        kps_norm = None

        results = yolo(frame, verbose=False, conf=0.3)[0]

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            confs = results.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            person = results.keypoints.data[best].cpu().numpy()

            valid_joints = person[:, 2] > VIDEO_CONFIG['conf_threshold']
            mean_conf = person[valid_joints, 2].mean() if valid_joints.any() else 0.0

            if person.shape == (17, 3) and mean_conf > 0.3:
                if last_valid_raw is not None:
                    torso_scale = np.linalg.norm(
                        person[8, :2] - person[0, :2]
                    ) if person[8, 2] > 0.3 else 1.0

                    for j in range(17):
                        if person[j, 2] < VIDEO_CONFIG['conf_threshold']:
                            person[j, :2] = last_valid_raw[j, :2]
                            person[j, 2] = VIDEO_CONFIG['conf_threshold'] * 0.5

                h36m = coco_to_h36m(person)
                kps_norm = normalize_h36m(h36m)
                last_valid_raw = person.copy()
                last_valid_norm = kps_norm.copy()

                draw_coco_skeleton(frame, person, VIDEO_CONFIG['conf_threshold'])

        if kps_norm is None:
            kps_norm = last_valid_norm.copy() if last_valid_norm is not None \
                       else np.zeros((17, 3), dtype=np.float32)

        all_kps.append(kps_norm.astype(np.float32))
        out_2d.write(frame)

    cap.release()
    out_2d.release()

    kps_array = np.array(all_kps)
    
    kps_array = smooth_2d_keypoints(
        kps_array, 
        VIDEO_CONFIG.get('2d_window_length', 11), 
        VIDEO_CONFIG.get('2d_polyorder', 3)
    )
    
    np.save('keypoints_2d.npy', kps_array)
    print(f"2D keypoint extraction done — shape: {kps_array.shape}")
    return kps_array, fps, (W, H)
