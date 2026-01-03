# ⚽ SoccerNet: Player Detection, Tracking and Behavior Analysis

**Artificial Vision Project Work 2025/2026** *University of Salerno - Dept. of Information Engineering, Electrical Engineering and Applied Mathematics*

**Authors:** Antonio Apicella, Antonio Graziosi

**Group:** 16

---

## 📖 Introduction

This project presents a complete Computer Vision pipeline developed for the **SoccerNet Video Understanding Benchmark Suite**. The goal is to address two main tasks on real soccer match video clips:

1. **Player Detection & Tracking:** Uniquely detect and track players and referees on the field ("persons of interest"), maintaining consistent identity despite occlusions, rapid camera movements, and the absence of distinctive visual features (identical jerseys).
2. **Behavior Analysis:** Estimate player density in specific *Regions of Interest* (ROI) of the field for each frame.

The proposed solution is based on an optimized **Tracking-by-Detection** approach, which combines a state-of-the-art detector (**YOLOv11x**) configured for high sensitivity (*High Recall*), an adaptive **Field Masking** module for removing false positives on the sidelines, and a motion-based tracker (**BoT-SORT** with global camera compensation GMC), devoid of visual Re-Identification modules.

The system was evaluated on the Challenge Test Set, achieving a PTBS (*Player Tracking and Behavior Score*) of **1.527**, with an HOTA of **0.742**.

---

## 🚀 Main Features

* **High-Recall Detector:** Use of **YOLOv11x** with 1088px input resolution and minimal confidence thresholds (0.1/0.2) to detect small objects and players blurred by motion.
* **Adaptive Field Masking:** Semantic segmentation algorithm combining HSV and LAB color spaces to dynamically identify the pitch and filter out stewards, photographers, and the audience, drastically reducing False Positives.
* **"Pure Motion" Tracking:** Implementation of **BoT-SORT** with *Sparse Optical Flow* (GMC) for camera motion compensation. The system does not use visual Re-ID, relying on robust geometric logic (`match_thresh: 0.9`) to avoid identity swaps between teammates.
* **Behavior Analysis:** Presence estimation in ROIs based on the geometric projection of the players' "feet point" (ground contact point).

---

## 📂 Repository Structure

```text
├── configs/              # YAML and JSON configuration files (tracker, ROI, main)
├── models/               # Model weights (YOLO, ReID) and conversion scripts
├── output/               # Output folder for logs, videos, and results
├── SIMULATOR/            # Code for simulation and local testing
├── src/                  # Main source code
│   ├── behaviour/        # Module for behavior analysis
│   ├── evaluation/       # Evaluation module (HOTA, nMAE) using TrackEval
│   ├── tracker/          # Tracker logic (YOLO + BoT-SORT integration)
│   ├── utils/            # Utilities (Field Masking, BBox operations, visualization)
│   ├── visualizer/       # Video generation with overlay
│   └── main.py           # Pipeline entry point
└── tracking/             # Dataset (train, test, challenge)

```

---

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/apiantonio/SoccerNet-tracking_AV2025-26
cd SoccerNet-tracking_AV2025-26

```


2. **Install dependencies:**
It is recommended to use a virtual environment (venv or conda).
```bash
pip install -r requirements.txt

```


*Main dependencies:* `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `scikit-image`, `pandas`, `trackeval`.

3. **Data Setup:**
Place the SoccerNet dataset in the `tracking/` folder. The expected structure for each sequence is:
```text
tracking/test/SNMOT-XXX/
├── img1/          # JPEG Frames
├── gt/gt.txt      # Ground Truth (optional for inference)
└── gameinfo.ini   # Metadata

```



---

## 💻 Usage

The pipeline can be executed via the `src/main.py` script. You can specify which steps to perform (tracking, analysis, visualization, evaluation).

### Basic Example (Full Execution)

```bash
python src/main.py --config configs/main_config.yaml --step all --seq all

```

### Main Arguments

* `--step`: Choose between `tracking`, `behaviour`, `eval`, `visualizer` or `all`.
* `--seq`: Specify one or more sequences (e.g., `SNMOT-116`) or `all` to process the entire folder.
* `--debug`: Enables on-screen visualization (`show_tracks`, `show_mask`, `show_behaviour`).

### Parameter Configuration

Tracker parameters can be modified in the `configs/botsort_8.yaml` file or passed via command line:

```bash
python src/main.py --conf 0.1 --iou 0.7 --tracker_config configs/botsort.yaml

```

---

## 📊 Results and Metrics

The system was evaluated on the official Test Set. Below are the obtained results:

| Metric | Value | Description |
| --- | --- | --- |
| **HOTA** | **0.742** | Higher Order Tracking Accuracy (Detection/Association Balance) |
| **DetA** | **0.863** | Detection Accuracy (Detection Precision) |
| **AssA** | **0.636** | Association Accuracy (Trajectory Stability) |
| **nMAE** | **0.785** | Normalized Mean Absolute Error (ROI Counting Precision) |
| **PTBS** | **1.527** | **Player Tracking and Behavior Score (Final Score)** |

### Strengths

* **High DetA (0.863):** Thanks to the "High Recall" strategy, the system almost never misses players, even in difficult situations.
* **Competitive AssA (0.636):** Despite the absence of Re-ID, the aggressive use of GMC and tight matching thresholds ensures stable tracking.

---

## 📜 References

* **YOLO11:** Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11*. [GitHub](https://github.com/ultralytics/ultralytics).
* **SoccerNet:** Deliege, A., et al. (2021). *SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos*.
* **BoT-SORT:** Aharon, N., et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*.

---

*Project developed for the Artificial Vision course, A.Y. 2025/2026.*
