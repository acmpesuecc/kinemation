# 3D pose rendering and visualization module

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import LEFT_JOINTS, RIGHT_JOINTS, RENDER_CONFIG


class Pose3DRenderer:
    # 3D pose renderer with matplotlib
    
    def __init__(self, size=None):
        if size is None:
            size = RENDER_CONFIG['panel_size']
            
        self.size = size
        self.dpi = 100
        self.elev = RENDER_CONFIG['elev']
        self.azim = RENDER_CONFIG['azim']
        self.fig = plt.figure(figsize=(size / self.dpi, size / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.patch.set_facecolor('#0d0d1a')

    def _bone_color(self, a, b):
        # Get color for bone based on joint indices
        if a in LEFT_JOINTS or b in LEFT_JOINTS:
            return '#00e676'
        if a in RIGHT_JOINTS or b in RIGHT_JOINTS:
            return '#ff4466'
        return '#90a4ae'

    def render(self, pose_3d, frame_num=0):
        # Render 3D pose to image
        self.ax.cla()
        self.ax.set_facecolor('#0d0d1a')
        self.fig.patch.set_facecolor('#0d0d1a')

        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#1a1a33')

        p = pose_3d.copy()
        
        x = p[:, 0]
        y = p[:, 1]
        z = -p[:, 2]

        p_disp = np.stack([x, y, z], axis=1)

        span = max(np.abs(p_disp).max() * RENDER_CONFIG['span_multiplier'], RENDER_CONFIG['min_span'])
        self.ax.set_xlim(-span, span)
        self.ax.set_ylim(-span, span)
        self.ax.set_zlim(-span, span)
        self.ax.view_init(elev=self.elev, azim=self.azim)

        h36m_connections = [
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6),
            (0,7), (7,8), (8,9), (9,10),
            (8,11), (11,12), (12,13),
            (8,14), (14,15), (15,16),
            (11,14),
        ]
        
        for a, b in h36m_connections:
            self.ax.plot(
                [p_disp[a,0], p_disp[b,0]],
                [p_disp[a,1], p_disp[b,1]],
                [p_disp[a,2], p_disp[b,2]],
                color=self._bone_color(a, b), linewidth=2
            )

        for j in range(17):
            c = '#ffcc00' if j == 0 else \
                '#00e676' if j in LEFT_JOINTS else \
                '#ff4466' if j in RIGHT_JOINTS else 'white'
            self.ax.scatter(
                [p_disp[j,0]], [p_disp[j,1]], [p_disp[j,2]],
                c=c, s=25, depthshade=False
            )

        pelvis = p_disp[0]
        spine = p_disp[8]
        spine_length = np.linalg.norm(spine - pelvis)

        self.ax.set_title(
            f'Frame {frame_num} | Spine Length: {spine_length:.3f}m',
            color='#00e676' if spine_length > 0.3 else '#ff4466',
            fontsize=8
        )
        
        self.ax.set_xlabel('X (L-R)', color='#666', fontsize=6)
        self.ax.set_ylabel('Y (U-D)', color='#666', fontsize=6) 
        self.ax.set_zlabel('Z (F-B)', color='#666', fontsize=6)
        
        self.ax.tick_params(colors='#333355', labelsize=5)
        self.ax.grid(True, color='#1a1a33', linewidth=0.3)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    def close(self):
        plt.close(self.fig)


def render_3d_video(video_path, poses_3d, fps, frame_size, output_path='output_3d.mp4'):
    # Render 3D poses to video with side-by-side layout
    W, H = frame_size
    P3D_SIZE = RENDER_CONFIG['panel_size']
    OUT_W = W + P3D_SIZE
    OUT_H = max(H, P3D_SIZE)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (OUT_W, OUT_H))
    renderer = Pose3DRenderer(size=P3D_SIZE)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(poses_3d):
            break

        frame = cv2.resize(frame, (W, H))
        panel = renderer.render(poses_3d[frame_idx], frame_num=frame_idx)

        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        canvas[:H, :W] = frame
        y_off = (OUT_H - P3D_SIZE) // 2
        canvas[y_off:y_off + P3D_SIZE, W:W + P3D_SIZE] = panel
        cv2.line(canvas, (W, 0), (W, OUT_H), (40, 40, 60), 1)

        out.write(canvas)
        frame_idx += 1

    cap.release()
    out.release()
    renderer.close()
    print(f"3D video rendering done — saved: {output_path}")
    return output_path
