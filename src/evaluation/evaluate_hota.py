import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter

# --- 1. FUNZIONI DI CALCOLO METRICA (HOTA Logic) ---

def calculate_iou_matrix(bbox_gt, bbox_pred):
    """
    Calcola matrice IoU tra GT e Pred.
    bbox_gt, bbox_pred: Array [N, 4] (x, y, w, h)
    """
    if len(bbox_gt) == 0 or len(bbox_pred) == 0:
        return np.zeros((len(bbox_gt), len(bbox_pred)))

    gt = np.expand_dims(bbox_gt, 1)    # [N, 1, 4]
    pr = np.expand_dims(bbox_pred, 0)  # [1, M, 4]

    gt_x1, gt_y1 = gt[..., 0], gt[..., 1]
    gt_x2, gt_y2 = gt_x1 + gt[..., 2], gt_y1 + gt[..., 3]
    
    pr_x1, pr_y1 = pr[..., 0], pr[..., 1]
    pr_x2, pr_y2 = pr_x1 + pr[..., 2], pr_y1 + pr[..., 3]

    xi1 = np.maximum(gt_x1, pr_x1)
    yi1 = np.maximum(gt_y1, pr_y1)
    xi2 = np.minimum(gt_x2, pr_x2)
    yi2 = np.minimum(gt_y2, pr_y2)
    
    inter_w = np.maximum(0, xi2 - xi1)
    inter_h = np.maximum(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    gt_area = gt[..., 2] * gt[..., 3]
    pr_area = pr[..., 2] * pr[..., 3]
    union_area = gt_area + pr_area - inter_area

    return inter_area / (union_area + 1e-6)

def compute_hota_05(gt_data, pred_data):
    """
    Calcola HOTA(0.5) per una singola sequenza.
    Input: Array Numpy [Frame, ID, X, Y, W, H]
    """
    gt_data = np.array(gt_data)
    pred_data = np.array(pred_data)
    
    # Se non ci sono dati
    if len(gt_data) == 0: return 0.0, 0.0, 0.0
    if len(pred_data) == 0: return 0.0, 0.0, 0.0

    frames = sorted(list(set(gt_data[:, 0]) | set(pred_data[:, 0])))
    
    TP_global = 0
    FN_global = 0
    FP_global = 0
    tp_matches = [] # Lista di tuple (gt_id, pred_id)

    # 1. Detection (Frame-by-Frame)
    for f in frames:
        gts = gt_data[gt_data[:, 0] == f]
        preds = pred_data[pred_data[:, 0] == f]
        
        if len(gts) == 0:
            FP_global += len(preds)
            continue
        if len(preds) == 0:
            FN_global += len(gts)
            continue

        iou_mat = calculate_iou_matrix(gts[:, 2:6], preds[:, 2:6])
        
        # Hungarian Algorithm
        cost_matrix = 1 - iou_mat
        # Soglia IoU < 0.5 diventa costo impossibile
        cost_matrix[iou_mat < 0.5] = 1000.0 
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_gt_inds = set()
        matched_pred_inds = set()
        
        for r, c in zip(row_ind, col_ind):
            # Controllo esplicito soglia IoU
            if iou_mat[r, c] >= 0.5:
                TP_global += 1
                tp_matches.append((int(gts[r, 1]), int(preds[c, 1])))
                matched_gt_inds.add(r)
                matched_pred_inds.add(c)
        
        FN_global += len(gts) - len(matched_gt_inds)
        FP_global += len(preds) - len(matched_pred_inds)

    if TP_global == 0:
        return 0.0, 0.0, 0.0

    # 2. Association (Global)
    # DetA = TP / (TP + FN + FP)
    DetA = TP_global / (TP_global + FN_global + FP_global)
    
    gt_counts = Counter(gt_data[:, 1].astype(int))
    pred_counts = Counter(pred_data[:, 1].astype(int))
    pair_counts = Counter(tp_matches)
    
    assa_sum = 0
    for (gt_id, pred_id) in tp_matches:
        tpa = pair_counts[(gt_id, pred_id)]
        fna = gt_counts[gt_id] - tpa
        fpa = pred_counts[pred_id] - tpa
        assa_sum += tpa / (tpa + fna + fpa)

    AssA = assa_sum / TP_global
    HOTA = np.sqrt(DetA * AssA)
    
    return HOTA, DetA, AssA

# --- 2. GESTIONE FILE I/O ---

def load_txt_6col(path):
    """
    Legge file txt [frame, id, x, y, w, h] saltando header/errori.
    Gestisce sia formati 6 colonne (Tracker) che 10 colonne (GT MOT).
    """
    data = []
    if not os.path.exists(path):
        print(f"ERROR: File not found {path}")
        return np.empty((0, 6))
        
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Supporta GT (10 col) prendendo solo le prime 6
            if len(parts) >= 6:
                try:
                    row = [float(x) for x in parts[:6]] 
                    data.append(row)
                except ValueError:
                    continue 
    return np.array(data) if len(data) > 0 else np.empty((0, 6))

def evaluate_sequences(base_gt_folder, tracker_output_folder, sequences, team_suffix="16"):
    """
    Calcola e visualizza i risultati HOTA.
    """
    hota_scores = []
    
    print("\n" + "="*60)
    print(f"{'SEQUENCE':<20} | {'HOTA(0.5)':<10} | {'DetA':<10} | {'AssA':<10}")
    print("-" * 60)
    
    for seq in sequences:
        # COSTRUZIONE PERCORSI:
        # 1. GT Path: assume struttura SoccerNet tracking/train/SNMOT-XXX/gt/gt.txt
        gt_path = os.path.join(base_gt_folder, seq, 'gt', 'gt.txt')
        
        # 2. Pred Path: assume output/tracking_SNMOT-XXX_16.txt
        pred_filename = f"tracking_{seq}_{team_suffix}.txt"
        pred_path = os.path.join(tracker_output_folder, pred_filename)

        # Controllo esistenza file
        if not os.path.exists(gt_path):
            print(f"{seq:<20} | GT MISSING ({gt_path})")
            continue
        if not os.path.exists(pred_path):
            print(f"{seq:<20} | PRED MISSING ({pred_path})")
            continue

        # Carica Dati
        gt_data = load_txt_6col(gt_path)
        pred_data = load_txt_6col(pred_path)
        
        # Calcola Metriche
        h, d, a = compute_hota_05(gt_data, pred_data)
        hota_scores.append(h)
        
        print(f"{seq:<20} | {h*100:6.2f} %   | {d*100:6.2f} %   | {a*100:6.2f} %")

    print("-" * 60)
    if hota_scores:
        avg_hota = np.mean(hota_scores)
        print(f"{'MEAN SCORE':<20} | {avg_hota * 100:6.2f} %   |")
    else:
        print("Nessuna sequenza valida valutata.")
    print("="*60 + "\n")

# --- MAIN ---

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    
    # Percorso base che CONTIENE le cartelle delle sequenze (es. SNMOT-060)
    # Assumendo che lanci lo script dalla root del progetto:
    GT_ROOT = "tracking/train"  
    
    # Cartella dove sono salvati i tuoi output .txt
    TRACKER_OUTPUT_DIR = "output\\buffer60_match70_high60_reid"
    
    # Suffisso del file di output (es. "_16" in tracking_SNMOT-060_16.txt)
    TEAM_SUFFIX = "16" 
    
    # Lista delle sequenze da valutare
    SEQUENCES_TO_EVAL = ["SNMOT-060"]
    
    # --- ESECUZIONE ---
    # Convertiamo i percorsi in assoluti per sicurezza
    base_dir = os.getcwd()
    gt_abs = os.path.join(base_dir, GT_ROOT)
    out_abs = os.path.join(base_dir, TRACKER_OUTPUT_DIR)
    
    print(f"Valutazione da: {base_dir}")
    print(f"GT Folder: {gt_abs}")
    print(f"Output Folder: {out_abs}")
    
    evaluate_sequences(gt_abs, out_abs, SEQUENCES_TO_EVAL, team_suffix=TEAM_SUFFIX)