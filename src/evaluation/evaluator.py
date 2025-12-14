import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter
from datetime import datetime
import json

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
    
    def _save_results_json(self, results_list, avg_hota, avg_nmae):
        """
        Salva un file JSON completo con Configurazione + Risultati.
        Questo file è perfetto per essere parsato automaticamente da script futuri.
        """
        # Nome del file univoco basato su timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Nome file es: results_20231025_143000.json
        filename = f"results_{timestamp}.json"
        save_path = os.path.join(self.output_folder, filename)
        
        # 2. Preparazione dei dati (Conversione tipi numpy in python native)
        # Calcolo PTBS (Somma HOTA + nMAE normalizzato, qui semplificato)
        ptbs = avg_hota + avg_nmae 

        structured_data = {
            "meta": {
                "timestamp": timestamp,
                "tracker_config_name": self.config['paths']['tracker_config'],
                "team_id": self.team_id
            },
            # Salviamo l'intera configurazione usata per questo run
            "configuration": self.config,
            
            # Metriche globali (facili da ordinare per trovare il migliore)
            "metrics_overall": {
                "HOTA_05": float(avg_hota),
                "nMAE": float(avg_nmae),
                "PTBS": float(ptbs)
            },
            
            # Metriche per singola sequenza
            "metrics_per_sequence": []
        }

        # Pulizia dati sequenze (conversione da numpy a float standard)
        for res in results_list:
            clean_res = {
                "seq": res['seq'],
                "hota": float(res['hota']),
                "deta": float(res['deta']),
                "assa": float(res['assa']),
                "nmae": float(res['nmae'])
            }
            structured_data["metrics_per_sequence"].append(clean_res)

        # 3. Scrittura su disco
        try:
            with open(save_path, 'w') as f:
                json.dump(structured_data, f, indent=4)
            print(f"💾 Report JSON strutturato salvato in: {save_path}")
        except Exception as e:
            print(f"⚠️ Errore salvataggio JSON: {e}")

    def _save_eval_results_in_track_cfg(self, results_list, avg_hota, avg_nmae):
        """
        Aggiunge i risultati di valutazione in coda al file di config del tracker.
        results_list: lista di dizionari {'seq': name, 'hota': val, ...}
        """
        # 1. Troviamo il nome del file config del tracker
        tracker_cfg_name = os.path.basename(self.config['paths']['tracker_config'])
        # 2. Costruiamo il percorso nella cartella output
        saved_cfg_path = os.path.join(self.output_folder, tracker_cfg_name)

        if os.path.exists(saved_cfg_path):
            try:
                with open(saved_cfg_path, 'a') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("\n\n# " + "=" * 79 + "\n")
                    f.write(f"# VALUTAZIONE AUTOMATICA - {timestamp}\n")
                    f.write(f'# imgsz: {self.config["tracker"]["imgsz"]}, device: {self.config["tracker"]["device"]}, half: {self.config["tracker"].get("half", False)}')
                    f.write(f"# " + "-" * 79 + "\n")
                    f.write(f"# {'SEQUENCE':<20} | {'HOTA(0.5)':<10} | {'DetA':<10} | {'AssA':<10} | {'nMAE':<10}\n")
                    f.write(f"# {'-' * 79}\n")
                    
                    # Scriviamo ogni singola sequenza
                    for res in results_list:
                        s_name = res['seq']
                        h = res['hota'] * 100
                        d = res['deta'] * 100
                        a = res['assa'] * 100
                        nm = res['nmae'] # Assumiamo sia già formattato o float
                        
                        f.write(f"# {s_name:<20} | {h:6.2f} %   | {d:6.2f} %   | {a:6.2f} %   | {nm} (TBD)\n")
                    
                    f.write(f"# {'-' * 79}\n")
                    f.write(f"# {'MEAN SCORES':<20} | {avg_hota * 100:6.2f} %   | {'-':<10} | {'-':<10} | {avg_nmae:<10}\n")
                    f.write("# " + "=" * 80 + "\n")
                
                print(f"📝 Report completo aggiunto a: {saved_cfg_path}")
            except Exception as e:
                print(f"⚠️ Errore scrittura metriche su file config: {e}")
        else:
            print(f"ℹ️ File config tracker non trovato in output ({saved_cfg_path}). Nessun log scritto.")

    def _compute_nmae(self, seq):
        """
        PLACEHOLDER: Calcolo nMAE per il behavior analysis.
        Da implementare in futuro leggendo il GT behavior e il file output behavior.
        
        TODO: Quando vorrai implementare il behavior scoring:
              Implementa il metodo _compute_nmae(self, seq) in evaluator.py.
              Dovrai caricare il file behavior_K_XX.txt e confrontarlo con un GT di behaviour (che dovrai avere o parsare dai dati XML/JSON del dataset).
              Infine aggiorna la stampa finale per calcolare PTBS = HOTA + nMAE (ricordando la formula di normalizzazione del nMAE dalla slide 10).
        """
        return 0.0 # Ritorna 0.0 (Non Definito) per ora

    def evaluate(self, sequences):
        """Esegue la pipeline di valutazione completa."""
        print("\n" + "="*80)
        print(f"{'SEQUENCE':<20} | {'HOTA(0.5)':<10} | {'DetA':<10} | {'AssA':<10} | {'nMAE':<10}")
        print("-" * 80)
        
        sequence_results = [] # Lista per accumulare i dati di ogni sequenza
        hota_values = []
        nmae_values = []

        for seq in sequences:
            # 1. Definizione Percorsi
            gt_path = os.path.join(self.input_folder, seq, 'gt', 'gt.txt')
            pred_path = os.path.join(self.output_folder, f"tracking_{seq}_{self.team_id}.txt")

            if not os.path.exists(gt_path):
                print(f"{seq:<20} | ⚠️ GT MISSING")
                continue
            
            if not os.path.exists(pred_path):
                print(f"{seq:<20} | ⚠️ PRED MISSING")
                continue

            # 2. Tracking Eval (HOTA)
            gt_data = self._load_txt(gt_path)
            pred_data = self._load_txt(pred_path)
            
            hota, deta, assa = self._compute_hota_05(gt_data, pred_data)
            
            # 3. Behaviour Eval (nMAE) - Placeholder
            nmae = self._compute_nmae(seq)

            # Accumulo valori per le medie
            hota_values.append(hota)
            nmae_values.append(nmae)

            # Accumulo dati per il report finale
            sequence_results.append({
                'seq': seq,
                'hota': hota,
                'deta': deta,
                'assa': assa,
                'nmae': nmae
            })

            # Stampa riga a console
            print(f"{seq:<20} | {hota*100:6.2f} %   | {deta*100:6.2f} %   | {assa*100:6.2f} %   | {nmae} (TBD)")

        print("-" * 80)
        
        avg_hota = np.mean(hota_values) if hota_values else 0.0
        avg_nmae = np.mean(nmae_values) if nmae_values else 0.0
        
        print(f"{'MEAN SCORES':<20} | {avg_hota * 100:6.2f} %   | {'-':<10} | {'-':<10} | {avg_nmae:<10}")
        print("="*80 + "\n")
        
        # Salvataggio risultati COMPLETI in coda al file di config del tracker
        if sequence_results:
            self._save_eval_results_in_track_cfg(sequence_results, avg_hota, avg_nmae)
            self._save_results_json(sequence_results, avg_hota, avg_nmae)