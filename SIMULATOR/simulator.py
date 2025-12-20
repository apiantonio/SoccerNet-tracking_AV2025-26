import os
import glob
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv

import cv2
import numpy as np
# ---- NumPy 2.x compatibility for TrackEval (older code uses deprecated aliases) ----
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from evaluation_helper import compute_nmae_from_behavior_files, compute_hota_at_05_trackeval, build_trackeval_structure, list_video_ids, natural_key

# ============================================================
# IO helpers for MOT-style tracking text files (10 columns)
# ============================================================

def read_behavior_per_frame(path: str) -> Dict[int, Dict[int, int]]:
    """
    Reads behavior file lines: frame, roi_id, count
    Returns: frame -> {roi_id: count}
    """
    per_frame: Dict[int, Dict[int, int]] = defaultdict(dict)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            fr = int(row[0])
            rid = int(row[1])
            n = int(float(row[2]))
            per_frame[fr][rid] = n
    return dict(per_frame)

def read_mot_txt_per_frame(txt_path: str) -> Dict[int, List[Tuple[int, int, int, int, int]]]:
    """
    Reads a MOT-style txt (CSV):
      frame, id, x, y, w, h, conf, class, vis, ...
    Returns per_frame dict:
      frame -> [(id, x, y, w, h), ...]
    """
    per_frame = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            fr = int(float(parts[0]))
            oid = int(float(parts[1]))
            x = int(float(parts[2]))
            y = int(float(parts[3]))
            w = int(float(parts[4]))
            h = int(float(parts[5]))
            per_frame[fr].append((oid, x, y, w, h))
    return dict(per_frame)


# ============================================================
# ROI + drawing helpers (optional; uses your roi.json format)
# ============================================================

def draw_roi_counts(
    img,
    rois: dict,
    roi_colors: dict,
    counts_for_frame: Dict[int, int],
    roi_ids: Tuple[int, int] = (1, 2),
):
    H, W = img.shape[:2]
    roi_names = sorted(rois.keys())

    for k, roi_name in enumerate(roi_names[:len(roi_ids)]):
        rid = roi_ids[k]
        val = int(counts_for_frame.get(rid, 0))

        r = rois[roi_name]
        x1, y1, x2, y2 = roi_pixel_rect(r, W, H)
        x1, y1, x2, y2 = clip_rect(x1, y1, x2, y2, W, H)

        # place text inside ROI (top-left with padding)
        tx = x1 + 10
        ty = y1 + 70

        color = roi_colors.get(roi_name, (255, 255, 255))

        # bigger number
        cv2.putText(
            img,
            str(val),
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,   # font scale (bigger)
            color,
            6,     # thickness (bolder)
            cv2.LINE_AA,
        )

def load_rois(roi_json_path: str) -> dict:
    with open(roi_json_path, "r") as f:
        return json.load(f)


def roi_pixel_rect(roi_norm: dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1 = int(round(roi_norm["x"] * img_w))
    y1 = int(round(roi_norm["y"] * img_h))
    rw = int(round(roi_norm["width"] * img_w))
    rh = int(round(roi_norm["height"] * img_h))
    return x1, y1, x1 + rw, y1 + rh


def clip_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def draw_rois(img, rois: dict, roi_colors: Optional[dict] = None):
    roi_colors = roi_colors or {}
    H, W = img.shape[:2]
    for name, r in rois.items():
        x1, y1, x2, y2 = roi_pixel_rect(r, W, H)
        x1, y1, x2, y2 = clip_rect(x1, y1, x2, y2, W, H)
        color = roi_colors.get(name, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_boxes_with_ids(img, items, color=(0, 255, 255), draw_ids=True):
    H, W = img.shape[:2]
    for (obj_id, x, y, bw, bh) in items:
        x1, y1, x2, y2 = clip_rect(x, y, x + bw, y + bh, W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if draw_ids:
            cv2.putText(img, str(obj_id), (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def bbox_base_center(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    # base-center = (center_x, bottom_y)
    return int(round(x + w / 2.0)), int(round(y + h))

def point_in_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_boxes_roi_colored(
    img,
    items,
    rois: dict,
    roi_colors: dict,
    out_color=(0, 255, 255),   # yellow in BGR
    draw_ids=True
):
    """
    Boxes inside an ROI (base-center point in ROI) take the ROI color.
    Boxes outside all ROIs are yellow (out_color).
    """
    H, W = img.shape[:2]
    roi_rects = [(name, roi_pixel_rect(r, W, H)) for name, r in rois.items()]

    for (obj_id, x, y, bw, bh) in items:
        cx, cy = bbox_base_center(x, y, bw, bh)

        roi_name = None
        for name, rect in roi_rects: # for this logic if a box is inside multiple rois, it takes the first one found roi name and color
            if point_in_rect(cx, cy, rect):
                roi_name = name
                break

        color = roi_colors.get(roi_name, out_color) if roi_name else out_color

        x1, y1, x2, y2 = clip_rect(x, y, x + bw, y + bh, W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (max(0, min(W - 1, cx)), max(0, min(H - 1, cy))), 3, color, -1)
        if draw_ids:
            cv2.putText(img, str(obj_id), (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ============================================================
# Interactive simulation viewer + final TrackEval HOTA@0.50
# ============================================================

def simulate_and_eval_hota05(
    dataset_root: str,
    predictions_root: str,
    group: str,
    fps: float = 25.0,
    split: str = "test",
    window_name: str = "GT (left) | PRED (right)",
    tmp_root: str = "./_tmp_trackeval_soccernet",
    show_window: bool = True,
):
    video_ids = list_video_ids(dataset_root)
    if not video_ids:
        raise FileNotFoundError(f"No numbered video folders found in: {dataset_root}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if show_window:
        ###################################################
        ########## Interactive simulation viewer ##########
        ###################################################
        for vid in video_ids:
            vid_dir = os.path.join(dataset_root, vid)
            frames_folder = os.path.join(vid_dir, "img1")
            gt_track_path = os.path.join(vid_dir, "gt", "gt.txt")
            roi_json_path = os.path.join(vid_dir, "roi.json")  # optional
            pred_track_path = os.path.join(predictions_root, f"tracking_{vid}_{group}.txt")

            gt_beh_path = os.path.join(vid_dir, "gt", "behavior_gt.txt")
            pr_beh_path = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")

            # create a structure that maps each frame each roi to the count of objects in that roi
            gt_beh = read_behavior_per_frame(gt_beh_path) if os.path.isfile(gt_beh_path) else {} # gt
            pr_beh = read_behavior_per_frame(pr_beh_path) if os.path.isfile(pr_beh_path) else {} # pred

            if not os.path.isfile(gt_track_path):
                raise FileNotFoundError(f"Missing GT gt.txt for video {vid}: {gt_track_path}")
            if not os.path.isfile(pred_track_path):
                raise FileNotFoundError(f"Missing prediction tracking for video {vid}: {pred_track_path}")

            rois = load_rois(roi_json_path) if os.path.isfile(roi_json_path) else None
            roi_colors = {}
            if rois is not None:
                # pick 2 rois deterministically: first = blue, second = red
                roi_names = sorted(rois.keys()) # rois.keys() = dict_keys(['roi1', 'roi2'])
                if len(roi_names) >= 1: # roi 1 = blue
                    roi_colors[roi_names[0]] = (255, 0, 0)   # blue (BGR)
                if len(roi_names) >= 2: # roi 2 = red
                    roi_colors[roi_names[1]] = (0, 0, 255)   # red (BGR)

            # map frame id -> [(onj id, x, y, w, h), ...]
            gt_per_frame = read_mot_txt_per_frame(gt_track_path) #gt
            pred_per_frame = read_mot_txt_per_frame(pred_track_path) #pred

            frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")), key=natural_key)
            if not frame_paths:
                raise FileNotFoundError(f"No jpg frames found for video {vid}: {frames_folder}")

            delay_ms = max(1, int(round(1000.0 / max(1e-6, fps)))) # number of milliseconds between frames to match fps (ms to wait between frames)
            paused = False
            i = 0

            while True:
                if not paused:
                    if i >= len(frame_paths): # end of video
                        break

                    frame = cv2.imread(frame_paths[i]) # read actual frame
                    if frame is None: # case of read error
                        i += 1
                        continue

                    frame_idx = i + 1  # starting from 1 

                    gt_list = gt_per_frame.get(frame_idx, []) # read the GT boxes for this frame
                    pr_list = pred_per_frame.get(frame_idx, []) # read the PRED boxes for this frame

                    left = frame.copy() # GT stream on left
                    right = frame.copy() # PRED stream on right

                    # --- Behavior overlay (GT left, Pred right) ---
                    gt_counts = gt_beh.get(frame_idx, {}) # read the  dict of roi_id -> count for this frame
                    pr_counts = pr_beh.get(frame_idx, {}) # read the  dict of roi_id -> count for this frame

                    # assuming roi ids are 1 and 2
                    gt_r1 = gt_counts.get(1, 0) # number of objects in roi 1 GT
                    gt_r2 = gt_counts.get(2, 0) # number of objects in roi 2 GT
                    pr_r1 = pr_counts.get(1, 0) # number of objects in roi 1 PRED
                    pr_r2 = pr_counts.get(2, 0) # number of objects in roi 2 PRED

                    gt_total = gt_r1 + gt_r2 # total number of objects GT in the rois
                    pr_total = pr_r1 + pr_r2 # total number of objects PRED in the rois

                    # draw total counts at bottom-left
                    cv2.putText(left,  f"TOTAL: {gt_total}", (10, left.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(right, f"TOTAL: {pr_total}", (10, right.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

                    # --- Draw ROIs, counts, boxes ---
                    if rois is not None:
                        # draw ROIs
                        draw_rois(left, rois, roi_colors)
                        draw_rois(right, rois, roi_colors)

                        # draw counts per roi at top-left read from behavior files
                        draw_roi_counts(left,  rois, roi_colors, gt_counts, roi_ids=(1, 2))
                        draw_roi_counts(right, rois, roi_colors, pr_counts, roi_ids=(1, 2))

                        # understand between all boxes which are inside which roi, and color roi colored boxes inside the roi and yellow those outside all rois
                        draw_boxes_roi_colored(left, gt_list, rois, roi_colors, out_color=(0, 255, 255), draw_ids=True)
                        draw_boxes_roi_colored(right, pr_list, rois, roi_colors, out_color=(0, 255, 255), draw_ids=True)
                    else:
                        # no ROI -> everything yellow
                        draw_boxes_with_ids(left, gt_list, color=(0, 255, 255), draw_ids=True)
                        draw_boxes_with_ids(right, pr_list, color=(0, 255, 255), draw_ids=True)

                    # frame info overlay
                    cv2.putText(left, f"GT | video {vid} | {group} | frame {frame_idx}/{len(frame_paths)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(right, f"PRED | video {vid} | {group} | frame {frame_idx}/{len(frame_paths)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    combined = cv2.hconcat([left, right]) # concatenate left and right images
                    cv2.line(combined, (left.shape[1], 0), (left.shape[1], combined.shape[0] - 1),
                            (255, 255, 255), 2) # vertical line in the middle

                    cv2.imshow(window_name, combined) # show the combined image
                    i += 1

                key = cv2.waitKey(delay_ms) & 0xFF # wait for key press with delay matching fps
                if key in (ord("q"), 27): # 'q' or 'Esc' to quit
                    cv2.destroyAllWindows()
                    return
                if key == ord(" "): # space to pause/play
                    paused = not paused
                if key == ord("n"): # 'n' for next video
                    break

            # End-of-video pause
            print(f"Finished video {vid}. Press 'n' for next, or 'q'/'Esc' to quit.")
            while True:
                key = cv2.waitKey(0) & 0xFF # wait indefinitely for key press
                if key in (ord("q"), 27): # 'q' or 'Esc' to quit
                    cv2.destroyAllWindows()
                    return
                if key == ord("n"): # 'n' for next video
                    break

        cv2.destroyAllWindows()
        ###################################################
        ###################################################
        ###################################################

    
    ##################################################
    ###### metrics computation with TrackEval ########
    ##################################################

    # --- TrackEval HOTA@0.50 on the whole dataset (official computation) ---

    # prepare TrackEval folder structure
    gt_folder, tr_folder, seqmap_file = build_trackeval_structure(
        dataset_root=dataset_root,
        predictions_root=predictions_root,
        group=group,
        split=split,
        fps=fps,
        tmp_root=tmp_root,
        benchmark="SNMOT",
        tracker_name="test",
    )

    # compute HOTA@0.50 using TrackEval
    hota_05 = compute_hota_at_05_trackeval(
        gt_folder=gt_folder,
        trackers_folder=tr_folder,
        seqmap_file=seqmap_file,
        split=split,
        benchmark="SNMOT",
        tracker_name="test",
    )

    shutil.rmtree(os.path.abspath(tmp_root), ignore_errors=True)
    print(f"\n=== TrackEval (SoccerNet-style) HOTA@0.50 ===\nHOTA@0.50: {hota_05:.6f}")

    print("\n--- Checking behavior files ---")
    missing = 0
    for vid in list_video_ids(dataset_root):
        gt_b = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_b = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")
        if not os.path.isfile(gt_b):
            print("MISSING GT behavior:", gt_b)
            missing += 1
        if not os.path.isfile(pr_b):
            print("MISSING PRED behavior:", pr_b)
            missing += 1
    print("Missing count:", missing)

    # --- Behavior metric (nMAE) ---
    beh = compute_nmae_from_behavior_files(
        dataset_root=dataset_root,
        predictions_root=predictions_root,
        group=group,
    )

    if beh.get("has_behavior", True) is False or beh.get("nMAE", None) is None:
        print("Behavior missing -> nMAE not computed (PTBS = HOTA@0.50)")
        print(f"PTBS: {hota_05:.6f}")
    else:
        print(f"MAE:  {beh['MAE']:.6f}")
        print(f"nMAE: {beh['nMAE']:.6f}")
        print(f"PTBS: {(hota_05 + beh['nMAE']):.6f}")


if __name__ == "__main__":
    simulate_and_eval_hota05(
        dataset_root="./tracking/test",
        predictions_root='./output/test_botsortprova1', #"./output/test_all_conf20_iou70_botsort8_reid",
        group="16",
        fps=200.0,
        split="test",
        show_window=False,
    )