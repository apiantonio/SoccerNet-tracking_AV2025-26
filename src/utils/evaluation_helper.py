import os
import shutil
import glob
import sys
import contextlib
from pathlib import Path
import cv2
import trackeval
from typing import Dict, Tuple, List
import numpy as np


# ============================================================
# Utility Generiche
# ============================================================

def natural_key(path: str) -> int:
    """Extracts numeric frame index from a file path for natural sorting."""
    name = os.path.basename(path)
    return int(name.split(".")[0])


def list_video_ids(dataset_root: str) -> List[str]:
    vids = []
    if not os.path.exists(dataset_root):
        return vids
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p) and name.isdigit():
            vids.append(name)
    return sorted(vids, key=lambda s: int(s))


@contextlib.contextmanager
def suppress_stdout():
    """Context manager per silenziare le stampe su console (stdout)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ============================================================
# Gestione Behaviour (nMAE)
# ============================================================

def _read_behavior(path: str) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}  # (frame, region_id) -> n_people
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
    Computes MAE and nMAE globally over all videos and both ROI ids.
    """
    abs_err_sum = 0.0
    n = 0
    has_all = True

    video_ids = list_video_ids(dataset_root)

    for vid in video_ids:
        gt_path = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_path = os.path.join(predictions_root, f"behavior_{vid}_{group}.txt")

        if not (os.path.isfile(gt_path) and os.path.isfile(pr_path)):
            has_all = False
            continue

        gt_b = _read_behavior(gt_path)
        pr_b = _read_behavior(pr_path)

        for key, gt_val in gt_b.items():
            pred_val = pr_b.get(key, 0)
            abs_err_sum += abs(pred_val - gt_val)
            n += 1

    if not has_all or n == 0:
        return {"has_behavior": False, "MAE": None, "nMAE": None}

    mae = abs_err_sum / n
    nmae = (10.0 - min(10.0, max(0.0, mae))) / 10.0
    return {"has_behavior": True, "MAE": mae, "nMAE": nmae}


# ============================================================
# Preparazione Dati per TrackEval
# ============================================================

def ensure_10col_and_force_class1(src_txt: str, dst_txt: str) -> None:
    """Writes a MOT 10-column file forcing class=1 (pedestrian)."""
    Path(dst_txt).parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    with open(src_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6: continue

            frame = parts[0]
            tid = parts[1]
            x, y, w, h = parts[2:6]
            conf = parts[6] if len(parts) >= 7 else "1"
            cls = "1"  # force pedestrian
            vis = parts[8] if len(parts) >= 9 else "-1"
            z = parts[9] if len(parts) >= 10 else "-1"

            out_lines.append(",".join([frame, tid, x, y, w, h, conf, cls, vis, z]))

    with open(dst_txt, "w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def write_seqinfo_ini(seq_dir: str, seq_name: str, fps: float, img_w: int, img_h: int, seq_len: int) -> None:
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


def build_trackeval_structure(
        dataset_root: str,
        predictions_root: str,
        group: str,
        split: str,
        fps: float,
        tmp_root: str,
        benchmark: str = "SNMOT",
        tracker_name: str = "test",
        target_sequences: List[str] = None
) -> Tuple[str, str, str]:
    """
    Creates the folder structure required by TrackEval.
    Returns: (gt_folder, trackers_folder, seqmap_file)
    """
    tmp_root = os.path.abspath(tmp_root)
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    gt_folder = os.path.join(tmp_root, "gt")
    tr_folder = os.path.join(tmp_root, "trackers")
    sm_folder = os.path.join(tmp_root, "seqmaps")
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(tr_folder, exist_ok=True)
    os.makedirs(sm_folder, exist_ok=True)

    bench_split = f"{benchmark}-{split}"
    gt_bs = os.path.join(gt_folder, bench_split)
    tr_bs = os.path.join(tr_folder, bench_split, tracker_name, "data")
    os.makedirs(gt_bs, exist_ok=True)
    os.makedirs(tr_bs, exist_ok=True)

    # Recupera tutte le sequenze
    all_seqs = list_video_ids(dataset_root)
    
    # Se abbiamo specificato sequenze target (e non è 'all'), filtriamo
    if target_sequences and 'all' not in target_sequences:
        seqs = [s for s in all_seqs if s in target_sequences]
    else:
        seqs = all_seqs

    if not seqs:
        raise FileNotFoundError(f"No numeric video folders found in: {dataset_root} matching the request.")
    
    for seq in seqs:
        src_seq = os.path.join(dataset_root, seq)
        src_img1 = os.path.join(src_seq, "img1")
        src_gt = os.path.join(src_seq, "gt", "gt.txt")
        src_pred = os.path.join(predictions_root, f"tracking_{seq}_{group}.txt")

        if not os.path.isfile(src_gt): raise FileNotFoundError(f"Missing GT: {src_gt}")
        if not os.path.isfile(src_pred): raise FileNotFoundError(f"Missing Pred: {src_pred}")

        frame_paths = sorted(glob.glob(os.path.join(src_img1, "*.jpg")), key=natural_key)
        if not frame_paths: raise FileNotFoundError(f"No frames in: {src_img1}")

        im0 = cv2.imread(frame_paths[0])
        H, W = im0.shape[:2]

        dst_seq = os.path.join(gt_bs, seq)
        os.makedirs(dst_seq, exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "gt"), exist_ok=True)
        os.makedirs(os.path.join(dst_seq, "img1"), exist_ok=True)

        write_seqinfo_ini(dst_seq, seq_name=seq, fps=fps, img_w=W, img_h=H, seq_len=len(frame_paths))
        ensure_10col_and_force_class1(src_gt, os.path.join(dst_seq, "gt", "gt.txt"))
        ensure_10col_and_force_class1(src_pred, os.path.join(tr_bs, f"{seq}.txt"))

    seqmap_file = os.path.join(sm_folder, f"{bench_split}.txt")
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in seqs:
            f.write(f"{seq}\n")

    return gt_folder, tr_folder, seqmap_file


# ============================================================
# TrackEval Esecuzione
# ============================================================

def compute_metrics_with_details(
        gt_folder: str,
        trackers_folder: str,
        seqmap_file: str,
        split: str,
        benchmark: str = "SNMOT",
        tracker_name: str = "test",
) -> List[Dict]:
    """
    Runs TrackEval silently and returns detailed metrics (HOTA, CLEAR) per sequence.
    """
    # Configurazione TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = False
    eval_config["PRINT_ONLY_COMBINED"] = False
    eval_config["PRINT_CONFIG"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "BENCHMARK": benchmark,
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": trackers_folder,
        "TRACKERS_TO_EVAL": [tracker_name],
        "SPLIT_TO_EVAL": split,
        "SEQMAP_FILE": seqmap_file,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "eval_results",
    })

    metrics_config = {"METRICS": ["HOTA", "CLEAR"]}

    # --- WRAPPING DI SILENZIAMENTO ---
    with suppress_stdout():
        try:
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

            metrics_list = []
            for metric in metrics_config['METRICS']:
                if metric == "HOTA":
                    metrics_list.append(trackeval.metrics.HOTA())
                elif metric == "CLEAR":
                    metrics_list.append(trackeval.metrics.CLEAR())

            output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
        except Exception:
            return []
    # ---------------------------------

    # Estrazione Dati
    hota_metric = trackeval.metrics.HOTA()
    alphas = np.array(hota_metric.array_labels, dtype=float)

    if "MotChallenge2DBox" not in output_res:
        return []

    try:
        idx_05 = int(np.where(np.isclose(alphas, 0.5))[0][0])
        tracker_data = output_res["MotChallenge2DBox"][tracker_name]
    except (KeyError, IndexError):
        return []

    detailed_results = []

    for seq_key in tracker_data.keys():
        try:
            hota_res = tracker_data[seq_key]["pedestrian"]["HOTA"]
            clear_res = tracker_data[seq_key]["pedestrian"]["CLEAR"]

            display_name = seq_key if seq_key != 'COMBINED_SEQ' else 'GLOBAL_SCORE'

            row = {
                'Video': display_name,
                'HOTA': round(float(hota_res['HOTA'][idx_05]), 6),
                'DetA': round(float(hota_res['DetA'][idx_05]), 6),
                'AssA': round(float(hota_res['AssA'][idx_05]), 6),
                'TP': int(clear_res['CLR_TP']),
                'FN': int(clear_res['CLR_FN']),
                'FP': int(clear_res['CLR_FP']),
            }
            detailed_results.append(row)
        except KeyError:
            continue

    def sort_key(x):
        if x['Video'] == 'GLOBAL_SCORE': return 999999
        try:
            return int(x['Video'])
        except:
            return 0

    detailed_results.sort(key=sort_key)
    return detailed_results


# Manteniamo la vecchia funzione per compatibilità, ma non è più necessaria
def compute_hota_at_05_trackeval(*args, **kwargs):
    print("Warning: compute_hota_at_05_trackeval is deprecated. Use compute_metrics_with_details.")
    return 0.0