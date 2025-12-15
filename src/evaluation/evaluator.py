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
        """Calcola HOTA(0.5) per una singola sequenza."""
        if len(gt_data) == 0: return 0.0, 0.0, 0.0
        if len(pred_data) == 0: return 0.0, 0.0, 0.0

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
            return 0.0, 0.0, 0.0

        # 2. Association (Global)
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
        Logica: un giocatore è nella ROI se il centro della base della bbox è dentro il rettangolo.
        """
        gt_behavior_path = os.path.join(self.input_folder, seq, 'gt', 'behavior_gt.txt')
        
        # Se esiste già, lo usiamo direttamente
        if os.path.exists(gt_behavior_path):
            return gt_behavior_path

        print(f"⚠️ Behavior GT mancante per {seq}. Generazione automatica in corso...")
        
        # 1. Recupero configurazione ROI
        roi_config_path = self.config['paths'].get('roi_config')
        if not roi_config_path or not os.path.exists(roi_config_path):
            # Fallback: prova a cercarlo nella cartella configs standard
            roi_config_path = os.path.join(os.getcwd(), 'configs', 'roi_config.json')
            
        if not os.path.exists(roi_config_path):
            print(f"❌ Configurazione ROI non trovata ({roi_config_path}). Impossibile generare GT.")
            return None

        with open(roi_config_path, 'r') as f:
            roi_data = json.load(f)

        # Risoluzione standard dataset SoccerNet
        IMG_W, IMG_H = 1920.0, 1080.0 

        # Helper: da relative (0-1) ad assolute (pixel)
        def get_abs_roi(r):
            return (r['x'] * IMG_W, r['y'] * IMG_H, 
                    (r['x'] + r['width']) * IMG_W, (r['y'] + r['height']) * IMG_H)

        roi1 = get_abs_roi(roi_data['roi1'])
        roi2 = get_abs_roi(roi_data['roi2'])

        # 2. Caricamento Tracking GT
        try:
            # Formato gt.txt: frame, id, x, y, w, h
            data = np.loadtxt(gt_tracking_path, delimiter=',')
        except Exception:
            return None

        if len(data) == 0:
            return None

        data = data.astype(float)
        unique_frames = np.unique(data[:, 0]).astype(int)
        
        lines_to_write = []
        
        # 3. Calcolo presenze per ogni frame
        for frame_idx in unique_frames:
            current_dets = data[data[:, 0] == frame_idx]
            
            c1, c2 = 0, 0
            for det in current_dets:
                # x, y, w, h sono agli indici 2, 3, 4, 5
                x, y, w, h = det[2], det[3], det[4], det[5]
                
                # Calcolo "Center of Basis"
                cx = x + w / 2.0
                cy = y + h
                
                # Check ROI 1
                if roi1[0] <= cx <= roi1[2] and roi1[1] <= cy <= roi1[3]:
                    c1 += 1
                
                # Check ROI 2
                if roi2[0] <= cx <= roi2[2] and roi2[1] <= cy <= roi2[3]:
                    c2 += 1
            
            # Formato output: frame_id, region_id, n_players
            lines_to_write.append(f"{frame_idx},1,{c1}\n")
            lines_to_write.append(f"{frame_idx},2,{c2}\n")

        # 4. Salvataggio su disco
        try:
            os.makedirs(os.path.dirname(gt_behavior_path), exist_ok=True)
            with open(gt_behavior_path, 'w') as f:
                f.writelines(lines_to_write)
            print(f"✅ Behavior GT generato: {gt_behavior_path}")
            return gt_behavior_path
        except Exception as e:
            print(f"❌ Errore salvataggio GT behavior: {e}")
            return None
    
    def _save_structured_json(self, results_list, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs):
        """
        Salva un file JSON completo con Configurazione Pipeline + Configurazione Tracker (effettiva) + Risultati.
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
            "meta": { "timestamp": timestamp, "tracker_config": tracker_cfg_name, "team_id": self.team_id },
            "main_config": self.config,
            "tracker_config": tracker_effective_config,
            "metrics_overall": {
                "HOTA_05": round(float(avg_hota), 4),
                "DetA": round(float(avg_deta), 4),
                "AssA": round(float(avg_assa), 4),
                "nMAE": round(float(global_nmae), 4), # Globale
                "PTBS": round(float(final_ptbs), 4),   # HOTA + Global nMAE
            },
            "metrics_per_sequence": results_list
        }
        
        try:
            with open(save_path, 'w') as f: json.dump(structured_data, f, indent=4)
            print(f"💾 Report JSON salvato: {save_path}")
        except Exception as e: print(f"⚠️ Errore JSON: {e}")

    def _save_eval_results_in_track_cfg(self, results_list, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs):
        """
        Aggiunge i risultati di valutazione in coda al file di config del tracker.
        results_list: lista di dizionari {'seq': name, 'hota': val, ...}
        """
        tracker_cfg_path = self.config['paths']['tracker_config']
        tracker_cfg_name = os.path.basename(tracker_cfg_path)
        saved_cfg_path = os.path.join(self.output_folder, tracker_cfg_name)

        if os.path.exists(saved_cfg_path):
            try:
                with open(saved_cfg_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("\n\n# " + "=" * 79 + "\n")
                    f.write(f"# VALUTAZIONE AUTOMATICA - {timestamp}\n")
                    f.write(f"# {'SEQUENCE':<20} | {'HOTA(0.5)':<10} | {'DetA':<10} | {'AssA':<10} | {'nMAE':<10}\n")
                    f.write(f"# {'-' * 79}\n")
                    
                    for res in results_list:
                        f.write(f"# {res['seq']:<20} | {res['hota']*100:6.2f} %   | {res['deta']*100:6.2f} %   | {res['assa']*100:6.2f} %   | {res['nmae']:.4f}\n")
                    
                    f.write(f"# {'-' * 79}\n")
                    f.write(f"# {'MEAN/GLOBAL SCORES':<20} | {avg_hota * 100:6.2f} %   | {avg_deta * 100:6.2f} %   | {avg_assa * 100:6.2f} %   | {global_nmae * 100:6.2f}\n")
                    f.write(f"# {'PTBS (HOTA + nMAE)':<20} | {final_ptbs:.4f}\n")
                    f.write("# " + "=" * 80 + "\n")
                print(f"📝 Report TXT aggiunto a: {saved_cfg_path}")
            except Exception as e: print(f"⚠️ Errore TXT: {e}")
            
    def _compute_raw_mae_data(self, seq):
        """
        Ritorna (total_absolute_error, num_samples) per una sequenza.
        Usato per calcolare sia il nMAE locale che contribuisce al globale.
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

    def evaluate(self, sequences):
        """Esegue la pipeline di valutazione completa."""
        print("\n" + "="*80)
        print(f"{'SEQUENCE':<20} | {'HOTA(0.5)':<10} | {'DetA':<10} | {'AssA':<10} | {'nMAE':<10}")
        print("-" * 80)
        
        sequence_results = [] 
        
        # Accumulatori per le medie
        hota_values = []
        deta_values = []
        assa_values = []
        
        # Accumulatori per nMAE GLOBALE
        global_abs_error = 0.0
        global_samples = 0

        for seq in sequences:
            # 1. Definizione Percorsi
            gt_path = os.path.join(self.input_folder, seq, 'gt', 'gt.txt')
            pred_path = os.path.join(self.output_folder, f"tracking_{seq}_{self.team_id}.txt")

            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                print(f"{seq:<20} | ⚠️ MISSING FILES")
                continue

            # 2. Tracking Eval (HOTA)
            gt_data = self._load_txt(gt_path)
            pred_data = self._load_txt(pred_path)
            hota, deta, assa = self._compute_hota_05(gt_data, pred_data)
            
            hota_values.append(hota)
            deta_values.append(deta)
            assa_values.append(assa)

            # 3. Behaviour Eval (nMAE)
            # Calcoliamo errore grezzo per il globale
            abs_err, samples = self._compute_raw_mae_data(seq)
            global_abs_error += abs_err
            global_samples += samples
            
            # Calcoliamo nMAE locale per il report per-sequenza
            local_mae = (abs_err / samples) if samples > 0 else 0.0
            local_nmae = self._normalize_mae(local_mae)

            # Accumulo dati per il report
            sequence_results.append({
                'seq': seq,
                'hota': hota,
                'deta': deta,
                'assa': assa,
                'nmae': local_nmae,
                'ptbs': hota + local_nmae # PTBS locale
            })

            print(f"{seq:<20} | {hota*100:6.2f} %   | {deta*100:6.2f} %   | {assa*100:6.2f} %   | {local_nmae:.4f}")

        print("-" * 80)
        
        # 4. Calcolo metriche FINALI
        avg_hota = np.mean(hota_values) if hota_values else 0.0
        avg_deta = np.mean(deta_values) if deta_values else 0.0
        avg_assa = np.mean(assa_values) if assa_values else 0.0
        
        # Calcolo nMAE GLOBALE (pesato su tutti i frame)
        if global_samples > 0:
            global_mae = global_abs_error / global_samples
            global_nmae = self._normalize_mae(global_mae)
        else:
            global_nmae = 0.0
            
        # Calcolo PTBS Finale
        final_ptbs = avg_hota + global_nmae
        
        print(f"{'MEAN/GLOBAL SCORES':<20} | {avg_hota * 100:6.2f} %   | {avg_deta * 100:6.2f} %   | {avg_assa * 100:6.2f} %   | {global_nmae:.4f}")
        print(f"{'PTBS (HOTA + nMAE)':<20} | {final_ptbs:6.4f}")
        print("="*80 + "\n")
        
        if sequence_results:
            self._save_eval_results_in_track_cfg(sequence_results, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs)
            self._save_structured_json(sequence_results, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs)
