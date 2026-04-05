# Kinemation

A computer vision pipeline for converting human movement in video into temporally coherent, smooth animated stick figures.

Kinemation detects body joints from monocular video, enforces temporal consistency across frames, and maps the resulting motion data to temporally coherent and smooth stick figure animations.

\---

## Table of Contents

* [Project Overview](#project-overview)
* [Pipeline Architecture](#pipeline-architecture)
* [Approaches Explored](#approaches-explored)

  * [Pose Estimation](#pose-estimation)
  * [Temporal Smoothing and Coherence](#temporal-smoothing-and-coherence)
* [Work Completed](#work-completed)
* [Work In Progress](#work-in-progress)
* [Setup and Installation](#setup-and-installation)
* [Team](#team)
* [References](#references)

\---

## Project Overview

Standard pose estimation models operate frame-by-frame, producing skeletal estimates that are accurate in isolation but temporally incoherent in sequence - joints flicker, limbs snap between positions, and the resulting animation is visually noisy. Kinemation addresses this by treating video as a temporal signal rather than a collection of independent images.

The longer-term goal extends beyond geometric accuracy. By integrating body-language-based emotion recognition, Kinemation aims to produce stick figures that not only move like the subject but also express like them - mapping inferred emotional state to visual properties of the animation such as posture, joint expressiveness, motion dynamics, and rendering style.

\---

## Pipeline Architecture

```
INPUT VIDEO
     |
     v
Person Detection (YOLOv8)
     |  Bounding boxes for each person
     v
Person Tracking (Hungarian Algorithm)
     |  Temporal ID assignment
     v
Preprocessing (CLAHE + Gaussian Blur)
     |  Enhanced frames per person
     v
2D Pose Estimator (MediaPipe BlazePose)
     |  N x 33 landmarks per frame per person
     v
Keypoint Adapter (MediaPipe -> H36M format)
     |  N x 17 x 2 keypoint sequence per person
     v
Temporal Smoothing (VideoPose3D TCN)
     |  Smooth 3D poses: N x 17 x 3 per person
     v
Dual Rendering Pipeline
     |
     +---> 2D Stick Figure (overlay on original)
     |
     +---> 3D Stick Figure (customizable background)
     |
     v
OUTPUT VIDEO (2D / 3D / Side-by-Side)
```

\---

## Approaches Explored

### Pose Estimation

The following methods were surveyed and evaluated for suitability in the Kinemation pipeline. Evaluation criteria included real-time performance, keypoint accuracy, ease of integration with downstream modules, and community support.

#### Tools and Libraries Evaluated

|Tool|Type|Keypoints|Status|
|-|-|-|-|
|MediaPipe BlazePose|2D/3D, real-time|33|In use|
|OpenPose|2D, bottom-up|25|Surveyed|
|OpenCV DNN|2D, lightweight|18|Surveyed|
|AlphaPose|2D, top-down|17/26|Surveyed|
|RTMPose|2D, real-time|17|Planned|
|HRNet|2D, top-down|17|Surveyed|
|ViTPose|2D, transformer|17|Surveyed|

#### Key Papers Reviewed

* Cao et al. (2017) - *Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields* (OpenPose) - [arXiv:1611.08050](https://arxiv.org/abs/1611.08050)
* Sun et al. (2019) - *Deep High-Resolution Representation Learning* (HRNet) - [arXiv:1908.07919](https://arxiv.org/abs/1908.07919)
* Newell et al. (2016) - *Stacked Hourglass Networks for Human Pose Estimation* - [arXiv:1603.06937](https://arxiv.org/abs/1603.06937)
* Yang et al. (2021) - *TransPose: Keypoint Localization via Transformer* - [arXiv:2012.14214](https://arxiv.org/abs/2012.14214)
* Xu et al. (2022) - *ViTPose: Simple Vision Transformer Baselines* - [arXiv:2204.12484](https://arxiv.org/abs/2204.12484)
* Jiang et al. (2023) - *RTMPose: Real-Time Multi-Person Pose Estimation* - [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
* Bazarevsky et al. (2020) - *BlazePose: On-device Real-time Body Pose Tracking* - [arXiv:2006.10204](https://arxiv.org/abs/2006.10204)

\---

### Temporal Smoothing and Coherence

Temporal coherence is the primary active research focus. Three papers were studied in depth, representing distinct approaches to the problem.

#### Paper 1 - Temporal Bundle Adjustment

Arnab, Doersch \& Zisserman (CVPR 2019) - *Exploiting Temporal Context for 3D Human Pose Estimation in the Wild* - [arXiv:1905.04668](https://arxiv.org/abs/1905.04668)

Treats the entire video as a single global optimization problem. Per-frame 2D keypoints are fed into an HMR model to produce SMPL mesh estimates (beta - body shape, theta - joint angles). Bundle Adjustment then jointly minimizes a compound loss E = E\_R + E\_T + E\_P across all frames simultaneously using L-BFGS. Body shape beta is held constant across frames to enforce anatomical consistency. Robust to noisy detections via Huber loss and camera shake via hinge loss.

* Strengths: globally consistent, principled, handles real-world noise well
* Limitations: requires full video upfront, computationally heavy, not suitable for real-time use

#### Paper 2 - Temporal Convolutional Network (Primary Implementation Target)

Pavllo et al. (CVPR 2019) - *3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training* - [arXiv:1811.11742](https://arxiv.org/abs/1811.11742)

Takes a sequence of 2D keypoints (T x J x 2) and uses a dilated 1D TCN with exponentially increasing dilation rates (1, 2, 4, 8...) to infer smooth 3D poses for the center frame of each window. Temporal smoothness is learned from data rather than explicitly optimized. Includes a semi-supervised extension using a back-projection loss, enabling training on unlabeled video without 3D ground truth. Inference is a single forward pass, making it significantly more practical than bundle adjustment for a real pipeline.

* Strengths: near-real-time, detector-agnostic, plug-and-play with MediaPipe outputs, semi-supervised capability
* Limitations: less globally consistent than bundle adjustment; receptive field of 243 frames requires padding on short clips

#### Paper 3 - Bidirectional 2D Temporal Refinement

Liu et al. (CVPR 2021) - *Deep Dual Consecutive Network for Human Pose Estimation* (DCPose) - [arXiv:2103.07254](https://arxiv.org/abs/2103.07254)

Operates purely in 2D. Takes a triplet of frames (t-k, t, t+k) and uses a Pose Temporal Merger (PTM) built on deformable convolutions to warp and align neighboring heatmaps to the target frame before merging. A Pose Refine Machine (PRM) then fuses the merged temporal features with the original single-frame heatmap to produce a corrected output. Occlusion recovery is an emergent property - joints hidden at frame t but visible at t+/-k are automatically recovered through the PTM-PRM pipeline without any explicit occlusion modeling.

* Strengths: stays in 2D, natural occlusion handling, architecturally elegant
* Limitations: operates on intermediate feature maps rather than final keypoint coordinates, making it non-trivial to integrate with arbitrary 2D detectors

#### Planned Upgrade - MotionBERT

Zhu et al. (ICCV 2023) - *MotionBERT: A Unified Perspective on Learning Human Motion Representations* - [arXiv:2210.06551](https://arxiv.org/abs/2210.06551)

Transformer-based temporal model accepting the same N x 17 x 2 input format as VideoPose3D, making it a straightforward upgrade once the base temporal pipeline is established. Demonstrates consistent accuracy improvements over TCN-based approaches, particularly on fast motion and occluded joints.

\---

## Work Completed

* Complete 3D pose estimation pipeline integrating MediaPipe BlazePose for 2D detection, YOLOv8 for person tracking, and VideoPose3D TCN for temporal 3D lifting
* Keypoint adapter successfully mapping MediaPipe's 33-landmark format to Human3.6M 17-joint format required by VideoPose3D
* Temporal smoothing implementation using VideoPose3D's dilated TCN with 243-frame receptive field
* Multi-person tracking with temporal consistency across frames using Hungarian algorithm for person-to-track assignment
* Advanced preprocessing pipeline including CLAHE enhancement in LAB color space and Gaussian blur for robust detection
* Dual visualization modes: 2D pose overlay on original video and 3D pose rendering with customizable backgrounds
* Web-based frontend interface built with Flask supporting both video upload and webcam capture
* Flexible output modes: 2D-only, 3D-only, or side-by-side comparison
* Comprehensive documentation of pipeline architecture and implementation details
* Literature survey covering 35+ papers across pose estimation paradigms including top-down, bottom-up, heatmap-based, regression-based, transformer-based, 3D, and video-based approaches
* In-depth study of three temporal smoothing papers covering global optimization, learned TCN-based smoothing, and bidirectional 2D refinement

\---

## Work In Progress

* Performance optimization for real-time webcam inference
* MotionBERT integration as an upgrade path for improved temporal coherence on fast motion and occluded joints
* Enhanced 3D visualization with interactive viewpoint manipulation
* Emotion recognition module for mapping inferred emotional state to visual animation properties

\---

## Setup and Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-org/kinemation
cd kinemation

# Install dependencies
pip install -r requirements.txt
```

### Required Models

Download the following models and place them in the `backend/models/` directory:

1. **YOLOv8 Nano Model**
   - Download: [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
   - Path: `backend/models/yolov8n.pt`

2. **MediaPipe Pose Landmarker**
   - Download: [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)
   - Path: `backend/models/pose_landmarker_lite.task`

3. **VideoPose3D Pretrained Checkpoint**
   - Download: [pretrained_h36m_detectron_coco.bin](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin)
   - Path: `backend/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin`

### Directory Structure

```
kinemation/
├── app.py
├── requirements.txt
├── backend/
│   ├── main.py
│   ├── mediapipe_to_h36m.py
│   ├── person_tracker.py
│   ├── visualizer_3d.py
│   ├── models/
│   │   ├── yolov8n.pt
│   │   └── pose_landmarker_lite.task
│   └── VideoPose3D/
│       └── checkpoint/
│           └── pretrained_h36m_detectron_coco.bin
└── frontend-resources/
    ├── templates/
    └── assets/
```

### Running the Application

#### Web Interface

```bash
python app.py
```

Access the web interface at `http://localhost:8080`

#### Command Line

```bash
# Process a video file
python backend/main.py --input path/to/video.mp4 --output path/to/output.mp4

# Choose output mode (2d_only, 3d_only, side_by_side)
python backend/main.py --input video.mp4 --output output.mp4 --mode side_by_side

# Customize 3D background color
python backend/main.py --input video.mp4 --output output.mp4 --bg-color "#1a1a1a"

# Process webcam input
python backend/main.py --webcam --output webcam_output.mp4
```

\---

## Team

**Mentor:** [Maaya Mohan](https://github.com/maayamohan)

**Mentees:**

* [Amrita Pradeep](https://github.com/amritap0200)
* [Hemaksh Breja](https://github.com/Hemaksh-b)
* [Kurias Joji](https://github.com/kurj17)
* [Navya S Gurupadmath](https://github.com/Navya2022)
* [Saisree Vaishnavi](https://github.com/svr-arch)
* [Yatin R](https://github.com/T1n777)

\---

## References

|Paper|Venue|Link|
|-|-|-|
|Cao et al. - *OpenPose*|CVPR 2017|[arXiv](https://arxiv.org/abs/1611.08050)|
|Newell et al. - *Stacked Hourglass Networks*|ECCV 2016|[arXiv](https://arxiv.org/abs/1603.06937)|
|Sun et al. - *HRNet*|CVPR 2019|[arXiv](https://arxiv.org/abs/1908.07919)|
|Xu et al. - *ViTPose*|NeurIPS 2022|[arXiv](https://arxiv.org/abs/2204.12484)|
|Jiang et al. - *RTMPose*|2023|[arXiv](https://arxiv.org/abs/2303.07399)|
|Bazarevsky et al. - *BlazePose*|2020|[arXiv](https://arxiv.org/abs/2006.10204)|
|Arnab, Doersch \& Zisserman - *Temporal Bundle Adjustment*|CVPR 2019|[arXiv](https://arxiv.org/abs/1905.04668)|
|Pavllo et al. - *VideoPose3D*|CVPR 2019|[arXiv](https://arxiv.org/abs/1811.11742)|
|Liu et al. - *DCPose*|CVPR 2021|[arXiv](https://arxiv.org/abs/2103.07254)|
|Zhu et al. - *MotionBERT*|ICCV 2023|[arXiv](https://arxiv.org/abs/2210.06551)|