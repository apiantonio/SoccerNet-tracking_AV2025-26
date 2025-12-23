import os, shutil, glob
from pathlib import Path
import cv2
import trackeval
from typing import Dict, Tuple, List
import numpy as np

def natural_key(path: str) -> int:
    """Extracts numeric frame index from a file path for natural sorting."""
    name = os.path.basename(path)
    return int(name.split(".")[0])

def _read_behavior(path: str) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {} # (frame, region_id) -> n_people
    import csv
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            fr = int(row[0])
            rid = int(row[1])
            n = int(float(row[2]))
            out[(fr, rid)] = n
    return out

def compute_nmae_from_behavior_files(dataset_root: str, predictions_root: str, group: str) -> dict:
    """
    Computes MAE and nMAE globally over all videos and both ROI ids
    using files:
      GT:   <video>/gt/behavior_gt.txt
      Pred: <predictions_root>/behavior_<video>_<group>.txt

    Missing prediction entries are treated as 0, like your old code.
    """
    # reuse your existing _read_behavior(path) which returns {(frame, roi): count}
    abs_err_sum = 0.0
    n = 0
    has_all = True

    video_ids = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and d.isdigit()] # get all the video ids
    video_ids.sort(key=lambda s: int(s)) # sort numerically

    for vid in video_ids:
        gt_path = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_path = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")

        if not (os.path.isfile(gt_path) and os.path.isfile(pr_path)):
            has_all = False
            continue

        # map (frame, region_id) -> n_people
        gt_b = _read_behavior(gt_path)
        pr_b = _read_behavior(pr_path)

        # Evaluate only where GT has an entry (same as your old code’s logic)
        for key, gt_val in gt_b.items(): # for each (frame, region_id) in GT
            pred_val = pr_b.get(key, 0) # select prediction associated to that key, default to 0 if missing
            abs_err_sum += abs(pred_val - gt_val) # accumulate absolute error
            n += 1

    if not has_all or n == 0: # if any video missing or no entries to evaluate
        return {"has_behavior": False, "MAE": None, "nMAE": None}

    mae = abs_err_sum / n
    nmae = (10.0 - min(10.0, max(0.0, mae))) / 10.0 # normalize MAE to [0,1] range as per metric definition
    return {"has_behavior": True, "MAE": mae, "nMAE": nmae}


# ============================================================
# Build TrackEval temp folders in MOTChallenge2DBox layout
# ============================================================


def ensure_10col_and_force_class1(src_txt: str, dst_txt: str) -> None:
    """
    Writes a MOT 10-column file:
      frame,id,x,y,w,h,conf,class,vis,unused
    Forces class=1 to be compatible with TrackEval's pedestrian-class evaluation.
    """
    Path(dst_txt).parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    with open(src_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame = parts[0]
            tid   = parts[1]
            x, y, w, h = parts[2:6]

            conf = parts[6] if len(parts) >= 7 else "1"
            cls  = "1"  # force pedestrian
            vis  = parts[8] if len(parts) >= 9 else "-1"
            z    = parts[9] if len(parts) >= 10 else "-1"

            out_lines.append(",".join([frame, tid, x, y, w, h, conf, cls, vis, z]))

    with open(dst_txt, "w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

def write_seqinfo_ini(seq_dir: str, seq_name: str, fps: float, img_w: int, img_h: int, seq_len: int) -> None:
    """
    Minimal seqinfo.ini for TrackEval MotChallenge2DBox, used also to compute SoccerNet Challenge metrics.
    it is used to give info about frame size, fps, length, etc. to TrackEval.
    it will write a seqinfo.ini file in seq_dir.
    """
    content = "\n".join([
        "[Sequence]",
        f"name={seq_name}",
        "imDir=img1",
        f"frameRate={int(round(fps))}",
        f"seqLength={int(seq_len)}",
        f"imWidth={int(img_w)}",
        f"imHeight={int(img_h)}",
        "imExt=.jpg",
        ""
    ])
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write(content)


def list_video_ids(dataset_root: str) -> List[str]:
    vids = []
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p) and name.isdigit():
            vids.append(name)
    return sorted(vids, key=lambda s: int(s))



# ============================================================
# TrackEval call: compute only HOTA@alpha=0.50
# ============================================================
def build_trackeval_structure(
    dataset_root: str,
    predictions_root: str,
    group: str,
    split: str,
    fps: float,
    tmp_root: str,
    benchmark: str = "SNMOT",
    tracker_name: str = "test",
) -> Tuple[str, str, str]:
    """
    Creates:
      tmp_root/
        gt/<BENCHMARK>-<SPLIT>/<SEQ>/gt/gt.txt
        gt/<BENCHMARK>-<SPLIT>/<SEQ>/seqinfo.ini
        trackers/<BENCHMARK>-<SPLIT>/<TRACKER>/data/<SEQ>.txt
        seqmaps/<BENCHMARK>-<SPLIT>.txt
    Returns:
      (gt_folder, trackers_folder, seqmap_file)
    """
    tmp_root = os.path.abspath(tmp_root) # make sure it's absolute
    if os.path.exists(tmp_root): # clean up in case it already exists
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True) # create base temp folder with proper format to compute metrics using TrackEval

    gt_folder = os.path.join(tmp_root, "gt") # will contain the ground-truth and images info in TrackEval format
    tr_folder = os.path.join(tmp_root, "trackers") # will contain the tracker predictions in TrackEval format
    sm_folder = os.path.join(tmp_root, "seqmaps") # will contain the series of sequences that we will evaluate
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(tr_folder, exist_ok=True)
    os.makedirs(sm_folder, exist_ok=True)

    bench_split = f"{benchmark}-{split}"
    gt_bs = os.path.join(gt_folder, bench_split)
    tr_bs = os.path.join(tr_folder, bench_split, tracker_name, "data")
    os.makedirs(gt_bs, exist_ok=True)
    os.makedirs(tr_bs, exist_ok=True)

    seqs = list_video_ids(dataset_root) # list of numeric folder names in dataset_root
    if not seqs:
        raise FileNotFoundError(f"No numeric video folders found in: {dataset_root}")

    for seq in seqs: # iterate over each numeric video folder
        src_seq = os.path.join(dataset_root, seq) # source sequence folder, e.g., #dataset_root/1
        src_img1 = os.path.join(src_seq, "img1") # source images folder, e.g., #dataset_root/1/img1
        src_gt = os.path.join(src_seq, "gt", "gt.txt") # source GT file, e.g., #dataset_root/1/gt/gt.txt
        src_pred = os.path.join(predictions_root, f"tracking_{seq}_{group}.txt") # source prediction file, e.g., #predictions_root/tracking_1_01.txt

        # check existence of required files
        if not os.path.isfile(src_gt): 
            raise FileNotFoundError(f"Missing GT gt.txt: {src_gt}")
        if not os.path.isfile(src_pred):
            raise FileNotFoundError(f"Missing prediction file: {src_pred}")

        frame_paths = sorted(glob.glob(os.path.join(src_img1, "*.jpg")), key=natural_key) # get all frames ordered by frame index
        if not frame_paths: # error in case no frames found
            raise FileNotFoundError(f"No frames in: {src_img1}")

        im0 = cv2.imread(frame_paths[0]) # read first frame to get dimensions, assuming all frames have same size
        if im0 is None:
            raise RuntimeError(f"Could not read: {frame_paths[0]}")
        H, W = im0.shape[:2] # get height and width
        seq_len = len(frame_paths) # number of frames in the sequence

        # Destination GT sequence folder
        dst_seq = os.path.join(gt_bs, seq)
        os.makedirs(dst_seq, exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "gt"), exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "img1"), exist_ok=True)  # can be empty; TrackEval uses seqinfo

        write_seqinfo_ini(dst_seq, seq_name=seq, fps=fps, img_w=W, img_h=H, seq_len=seq_len) # write seqinfo.ini file used by TrackEval

        # it rewrite the GT and prediction file in 10-column format to ensure compatibility, class forced to 1 (pedestrian)
        ensure_10col_and_force_class1(src_gt, os.path.join(dst_seq, "gt", "gt.txt")) 
        ensure_10col_and_force_class1(src_pred, os.path.join(tr_bs, f"{seq}.txt"))

    # Seqmap file (TrackEval expects first line header; then one seq=video per line) it is used to tell TrackEval which sequences to evaluate
    seqmap_file = os.path.join(sm_folder, f"{bench_split}.txt")
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in seqs:
            f.write(f"{seq}\n")

    return gt_folder, tr_folder, seqmap_file


def compute_hota_at_05_trackeval(
    gt_folder: str,
    trackers_folder: str,
    seqmap_file: str,
    split: str,
    benchmark: str = "SNMOT",
    tracker_name: str = "test",
) -> float:
    """
    Runs TrackEval MotChallenge2DBox with only HOTA metric family,
    then extracts HOTA at alpha=0.50 (not averaged over alphas).
    """
    # --- configs (no argparse; direct dicts) ---
    eval_config = trackeval.Evaluator.get_default_eval_config() # get default eval config, as done in SoccerNet Challenge
    eval_config["DISPLAY_LESS_PROGRESS"] = True # show less progress otherwise too verbose, set to False to see full progress

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config() # get default dataset config for MotChallenge2DBox, as done in SoccerNet Challenge
    dataset_config.update({
        "BENCHMARK": benchmark,
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": trackers_folder,
        "TRACKERS_TO_EVAL": [tracker_name],
        "SPLIT_TO_EVAL": split,
        "SEQMAP_FILE": seqmap_file,
        "DO_PREPROC": False,          # matches SoccerNet wrapper style
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "eval_results",
    }) # update with our specific paths and settings

    metrics_config = {"METRICS": ["HOTA"]} # only HOTA metric needed for the Challenge

    evaluator = trackeval.Evaluator(eval_config) # create evaluator with eval config
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)] # create dataset list with our dataset config
    metrics_list = [trackeval.metrics.HOTA(metrics_config)] # create metrics list with HOTA metric

    output_res, _ = evaluator.evaluate(dataset_list, metrics_list) # run official evaluation 

    # Find alpha index for 0.50 from the metric itself
    hota_metric = trackeval.metrics.HOTA(metrics_config) # create a HOTA curve metrics for an array of alphas
    alphas = np.array(hota_metric.array_labels, dtype=float) # get array of alphas used in HOTA curve
    idx = int(np.where(np.isclose(alphas, 0.5))[0][0]) # find index of alpha=0.50 

    # SoccerNet-style extraction uses COMBINED_SEQ and class key "pedestrian"
    hota_curve = output_res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]["HOTA"]["HOTA"] # from the output results, get HOTA curve for combined sequences and pedestrian class 
    return float(hota_curve[idx]) # return HOTA@0.50 value as float
