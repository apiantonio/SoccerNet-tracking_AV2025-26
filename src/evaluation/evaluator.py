import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter
from datetime import datetime
import json
import yaml

class Evaluator:
    def __init__(self, config):
        """
        Inizializza il valutatore.
        config: dizionario caricato da yaml
        """
        self.config = config
        self.input_folder = config['paths']['input_folder']   # Cartella dataset (dove cercare i GT)
        self.output_folder = config['paths']['output_folder'] # Cartella predizioni
        self.team_id = config['settings']['team_id']

    def _calculate_iou_matrix(self, bbox_gt, bbox_pred):
        """Calcola matrice IoU tra GT e Pred."""
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

    def _compute_hota_05(self, gt_data, pred_data):
        """Calcola HOTA(0.5) e metriche raw (TP, FN, FP, TPA, FNA, FPA) per una singola sequenza."""
        if len(gt_data) == 0: return 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0
        if len(pred_data) == 0: return 0.0, 0.0, 0.0, 0, len(gt_data), 0, 0, 0, 0

        frames = sorted(list(set(gt_data[:, 0]) | set(pred_data[:, 0])))
        
        TP_global = 0
        FN_global = 0
        FP_global = 0
        tp_matches = [] 

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

            iou_mat = self._calculate_iou_matrix(gts[:, 2:6], preds[:, 2:6])
            
            cost_matrix = 1 - iou_mat
            cost_matrix[iou_mat < 0.5] = 1000.0 
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_gt_inds = set()
            matched_pred_inds = set()
            
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= 0.5:
                    TP_global += 1
                    tp_matches.append((int(gts[r, 1]), int(preds[c, 1])))
                    matched_gt_inds.add(r)
                    matched_pred_inds.add(c)
            
            FN_global += len(gts) - len(matched_gt_inds)
            FP_global += len(preds) - len(matched_pred_inds)

        if TP_global == 0:
            return 0.0, 0.0, 0.0, 0, FN_global, FP_global, 0, 0, 0

        # 2. Association (Global)
        DetA = TP_global / (TP_global + FN_global + FP_global)
        
        gt_counts = Counter(gt_data[:, 1].astype(int))
        pred_counts = Counter(pred_data[:, 1].astype(int))
        pair_counts = Counter(tp_matches)
        
        assa_sum = 0
        
        # Accumulatori per metriche raw di associazione
        TPA_total = 0
        FNA_total = 0
        FPA_total = 0

        for (gt_id, pred_id) in tp_matches:
            tpa = pair_counts[(gt_id, pred_id)]
            fna = gt_counts[gt_id] - tpa
            fpa = pred_counts[pred_id] - tpa
            
            assa_sum += tpa / (tpa + fna + fpa)
            
            # Accumulo i raw counts (nota: questi sono sommati su tutti i match TP)
            TPA_total += tpa
            FNA_total += fna
            FPA_total += fpa

        AssA = assa_sum / TP_global
        HOTA = np.sqrt(DetA * AssA)
        
        return HOTA, DetA, AssA, TP_global, FN_global, FP_global, TPA_total, FNA_total, FPA_total

    def _load_txt(self, path):
        """Legge file txt [frame, id, x, y, w, h] ignorando colonne extra."""
        data = []
        if not os.path.exists(path):
            return np.empty((0, 6))
            
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        row = [float(x) for x in parts[:6]] 
                        data.append(row)
                    except ValueError:
                        continue 
        return np.array(data) if len(data) > 0 else np.empty((0, 6))

    def _load_behavior_file(self, path):
        """
        Carica il file behavior in un dizionario per accesso rapido.
        Return: dict {(frame_id, region_id): n_players}
        """
        data = {}
        if not os.path.exists(path):
            return data
            
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        frame_id = int(parts[0])
                        region_id = int(parts[1])
                        count = int(parts[2])
                        data[(frame_id, region_id)] = count
                    except ValueError:
                        continue
        return data

    def _generate_missing_behavior_gt(self, seq, gt_tracking_path):
        """
        Genera il file GT del behavior partendo dal GT del tracking se non esiste.
        """
        gt_behavior_path = os.path.join(self.input_folder, seq, 'gt', 'behavior_gt.txt')
        
        if os.path.exists(gt_behavior_path):
            return gt_behavior_path

        roi_config_path = self.config['paths'].get('roi_config')
        if not roi_config_path or not os.path.exists(roi_config_path):
            roi_config_path = os.path.join(os.getcwd(), 'configs', 'roi_config.json')
            
        if not os.path.exists(roi_config_path):
            print(f"❌ Configurazione ROI non trovata ({roi_config_path}). Impossibile generare GT.")
            return None

        with open(roi_config_path, 'r') as f:
            roi_data = json.load(f)

        IMG_W, IMG_H = 1920.0, 1080.0 

        def get_abs_roi(r):
            return (r['x'] * IMG_W, r['y'] * IMG_H, 
                    (r['x'] + r['width']) * IMG_W, (r['y'] + r['height']) * IMG_H)

        roi1 = get_abs_roi(roi_data['roi1'])
        roi2 = get_abs_roi(roi_data['roi2'])

        try:
            data = np.loadtxt(gt_tracking_path, delimiter=',')
        except Exception:
            return None

        if len(data) == 0:
            return None

        data = data.astype(float)
        unique_frames = np.unique(data[:, 0]).astype(int)
        
        lines_to_write = []
        
        for frame_idx in unique_frames:
            current_dets = data[data[:, 0] == frame_idx]
            
            c1, c2 = 0, 0
            for det in current_dets:
                x, y, w, h = det[2], det[3], det[4], det[5]
                cx = x + w / 2.0
                cy = y + h
                
                if roi1[0] <= cx <= roi1[2] and roi1[1] <= cy <= roi1[3]:
                    c1 += 1
                if roi2[0] <= cx <= roi2[2] and roi2[1] <= cy <= roi2[3]:
                    c2 += 1
            
            lines_to_write.append(f"{frame_idx},1,{c1}\n")
            lines_to_write.append(f"{frame_idx},2,{c2}\n")

        try:
            os.makedirs(os.path.dirname(gt_behavior_path), exist_ok=True)
            with open(gt_behavior_path, 'w') as f:
                f.writelines(lines_to_write)
            return gt_behavior_path
        except Exception as e:
            print(f"❌ Errore salvataggio GT behavior: {e}")
            return None
    
    def _save_structured_json(self, results_list, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs, global_raw_counts):
        """
        Salva un file JSON completo con Configurazione Pipeline + Configurazione Tracker + Risultati.
        """
        tracker_cfg_path = self.config['paths']['tracker_config']
        tracker_cfg_name = os.path.basename(tracker_cfg_path).replace('.yaml', '') if tracker_cfg_path else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{tracker_cfg_name}_{timestamp}.json"
        save_path = os.path.join(self.output_folder, filename)

        tracker_effective_config = {}
        if tracker_cfg_path and os.path.exists(tracker_cfg_path):
            try:
                with open(tracker_cfg_path, 'r') as f: tracker_effective_config = yaml.safe_load(f) or {}
            except: pass
        
        structured_data = {
            "meta": { 
                "timestamp": timestamp,
                "tracker_config": tracker_cfg_name, 
                "team_id": self.team_id 
            },
            
            "main_config": self.config,

            "tracker_config": tracker_effective_config,
            
            "metrics_overall": {
                "HOTA_05": round(float(avg_hota), 4),
                "DetA": round(float(avg_deta), 4),
                "AssA": round(float(avg_assa), 4),
                "nMAE": round(float(global_nmae), 4),
                "PTBS": round(float(final_ptbs), 4),
                "counts_sum": global_raw_counts # Totali globali di TP, FN, FP...
            },
            
            "metrics_per_sequence": results_list
        }
        
        try:
            with open(save_path, 'w') as f: json.dump(structured_data, f, indent=4)
            print(f"💾 Report JSON salvato: {save_path}")
        except Exception as e: 
            print(f"⚠️ Errore JSON: {e}")

    def _save_eval_results_in_track_cfg(self, results_list, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs, global_raw_counts):
        """
        Aggiunge i risultati di valutazione in coda al file di config del tracker.
        """
        tracker_cfg_path = self.config['paths']['tracker_config']
        tracker_cfg_name = os.path.basename(tracker_cfg_path)
        saved_cfg_path = os.path.join(self.output_folder, tracker_cfg_name)

        if os.path.exists(saved_cfg_path):
            try:
                with open(saved_cfg_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("\n\n# " + "=" * 129 + "\n")
                    f.write(f"# VALUTAZIONE AUTOMATICA - {timestamp}\n")
                    f.write(f"# {'SEQUENCE':<15} | {'HOTA':<7} | {'DetA':<7} | {'AssA':<7} | {'nMAE':<7} | {'TP':<6} | {'FN':<6} | {'FP':<6} | {'TPA':<7} | {'FNA':<7} | {'FPA':<7} |\n")
                    f.write(f"# {'-' * 129}\n")
                    
                    for res in results_list:
                        f.write(f"# {res['seq']:<15} | {res['hota']*100:6.2f}% | {res['deta']*100:6.2f}% | {res['assa']*100:6.2f}% | {res['nmae']*100:6.2f}% | {res['TP']:<6} | {res['FN']:<6} | {res['FP']:<6} | {res['TPA']:<7} | {res['FNA']:<7} | {res['FPA']:<7} |\n")
                    
                    f.write(f"# {'-' * 129}\n")
                    f.write(f"# {'GLOBAL/SUM':<15} | {avg_hota * 100:6.2f}% | {avg_deta * 100:6.2f}% | {avg_assa * 100:6.2f}% | {global_nmae * 100:6.2f}% | {global_raw_counts['TP']:<6} | {global_raw_counts['FN']:<6} | {global_raw_counts['FP']:<6} | {global_raw_counts['TPA']:<7} | {global_raw_counts['FNA']:<7} | {global_raw_counts['FPA']:<7} |\n")
                    f.write(f"# {'PTBS (HOTA + nMAE)':<20} | {final_ptbs:7.4f}\n")
                    f.write("# " + "=" * 130 + "\n")
                print(f"📝 Report TXT aggiunto a: {saved_cfg_path}")
            except Exception as e: print(f"⚠️ Errore TXT: {e}")
            
    def _compute_raw_mae_data(self, seq):
        """
        Ritorna (total_absolute_error, num_samples) per una sequenza.
        """
        gt_tracking_path = os.path.join(self.input_folder, seq, 'gt', 'gt.txt')
        gt_behavior_path = self._generate_missing_behavior_gt(seq, gt_tracking_path)
        pred_path = os.path.join(self.output_folder, f"behavior_{seq}_{self.team_id}.txt")

        if not gt_behavior_path or not os.path.exists(gt_behavior_path): return 0.0, 0
        if not os.path.exists(pred_path): return 0.0, 0

        gt_data = self._load_behavior_file(gt_behavior_path)
        pred_data = self._load_behavior_file(pred_path)
        
        if not gt_data: return 0.0, 0

        total_abs_error = 0.0
        n_samples = 0

        for key, gt_count in gt_data.items():
            pred_count = pred_data.get(key, 0)
            total_abs_error += abs(pred_count - gt_count)
            n_samples += 1
            
        return total_abs_error, n_samples

    def _normalize_mae(self, mae):
        """Formula di normalizzazione standard"""
        return (10.0 - min(10.0, mae)) / 10.0

    def _get_ignore_ids(self, seq):
        """
        Legge gameinfo.ini e restituisce un set di ID da ignorare (es. la palla).
        """
        ini_path = os.path.join(self.input_folder, seq, "gameinfo.ini")
        ignore_ids = set()
        
        if not os.path.exists(ini_path):
            return ignore_ids
        
        try:
            with open(ini_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("trackletID_") and "=" in line:
                        key, val = line.split("=", 1)
                        if "ball" in val.lower():
                            try:
                                track_id = int(key.split("_")[1])
                                ignore_ids.add(track_id)
                            except (IndexError, ValueError):
                                continue
        except Exception as e:
            print(f"⚠️ Errore lettura gameinfo.ini per {seq}: {e}")
            
        return ignore_ids

    def evaluate(self, sequences):
        """Esegue la pipeline di valutazione completa."""
        print("\n" + "="*130)
        # Intestazione allargata per includere i nuovi campi
        print(f"{'SEQUENCE':<15} | {'HOTA':<7} | {'DetA':<7} | {'AssA':<7} | {'nMAE':<7} | {'TP':<6} | {'FN':<6} | {'FP':<6} | {'TPA':<7} | {'FNA':<7} | {'FPA':<7} |")
        print("-" * 130)
        
        sequence_results = [] 
        
        # Accumulatori per le medie/metriche
        hota_values = []
        deta_values = []
        assa_values = []
        
        # Accumulatori per nMAE GLOBALE
        global_abs_error = 0.0
        global_samples = 0

        # Accumulatori per conteggi RAW GLOBALI
        global_raw = {
            'TP': 0, 'FN': 0, 'FP': 0,
            'TPA': 0, 'FNA': 0, 'FPA': 0
        }

        for seq in sequences:
            # 1. Definizione Percorsi
            gt_path = os.path.join(self.input_folder, seq, 'gt', 'gt.txt')
            pred_path = os.path.join(self.output_folder, f"tracking_{seq}_{self.team_id}.txt")

            if not os.path.exists(gt_path):
                print(f"{seq:<15} | ⚠️ MISSING GT: {gt_path}")
                continue
            
            if not os.path.exists(pred_path):
                print(f"{seq:<15} | ⚠️ MISSING PREDICTIONS: {pred_path}")
                continue

            # 2. Tracking Eval (HOTA + Raw Counts)
            gt_data = self._load_txt(gt_path)
            pred_data = self._load_txt(pred_path)
            
            # Filtro palla
            ignore_ids = self._get_ignore_ids(seq)
            if len(gt_data) > 0 and ignore_ids:
                mask = [int(row[1]) not in ignore_ids for row in gt_data]
                gt_data = gt_data[mask]
            
            if len(pred_data) == 0:
                print(f"{seq:<15} | ⚠️ EMPTY TRACKING (Skipping)")
                continue

            # Chiamata aggiornata a _compute_hota_05
            hota, deta, assa, tp, fn, fp, tpa, fna, fpa = self._compute_hota_05(gt_data, pred_data)
            
            hota_values.append(hota)
            deta_values.append(deta)
            assa_values.append(assa)
            
            # Aggiornamento conteggi globali
            global_raw['TP'] += tp
            global_raw['FN'] += fn
            global_raw['FP'] += fp
            global_raw['TPA'] += tpa
            global_raw['FNA'] += fna
            global_raw['FPA'] += fpa

            # 3. Behaviour Eval (nMAE)
            abs_err, samples = self._compute_raw_mae_data(seq)
            global_abs_error += abs_err
            global_samples += samples
            
            local_mae = (abs_err / samples) if samples > 0 else 0.0
            local_nmae = self._normalize_mae(local_mae)
            ptbs_local = hota + local_nmae

            # Accumulo dati per il report
            res_entry = {
                'seq':  seq,
                'hota': round(hota, 4),
                'deta': round(deta, 4),
                'assa': round(assa, 4),
                'nmae': round(local_nmae, 4),
                'ptbs': round(ptbs_local, 4),
                # Raw counts
                'TP': tp, 'FN': fn, 'FP': fp,
                'TPA': int(tpa), 'FNA': int(fna), 'FPA': int(fpa)
            }
            sequence_results.append(res_entry)

            # Stampa riga
            print(f"{seq:<15} | {hota*100:6.2f}% | {deta*100:6.2f}% | {assa*100:6.2f}% | {local_nmae*100:6.2f}% | {tp:<6} | {fn:<6} | {fp:<6} | {int(tpa):<7} | {int(fna):<7} | {int(fpa):<7} |")

        print("-" * 130)
        
        # 4. Calcolo metriche FINALI
        avg_hota = np.mean(hota_values) if hota_values else 0.0
        avg_deta = np.mean(deta_values) if deta_values else 0.0
        avg_assa = np.mean(assa_values) if assa_values else 0.0
        
        if global_samples > 0:
            global_mae = global_abs_error / global_samples
            global_nmae = self._normalize_mae(global_mae)
        else:
            global_nmae = 0.0
            
        final_ptbs = avg_hota + global_nmae
        
        # Stampa riga globale
        print(f"{'MEAN/GLOBAL':<15} | {avg_hota * 100:6.2f}% | {avg_deta * 100:6.2f}% | {avg_assa * 100:6.2f}% | {global_nmae * 100:6.2f}% | {global_raw['TP']:<6} | {global_raw['FN']:<6} | {global_raw['FP']:<6} | {global_raw['TPA']:<7} | {global_raw['FNA']:<7} | {global_raw['FPA']:<7} |")
        print(f"{'PTBS':<15} | {final_ptbs:7.4f}")
        print("="*130 + "\n")
        
        if sequence_results:
            self._save_eval_results_in_track_cfg(sequence_results, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs, global_raw)
            self._save_structured_json(sequence_results, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs, global_raw)