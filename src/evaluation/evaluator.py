import os
import shutil
import json
import numpy as np
from datetime import datetime
import yaml
from utils.bbox_operations import BBoxOperations
from utils.evaluation_helper import build_trackeval_structure, compute_metrics_with_details, compute_nmae_from_behavior_files
# ---- NumPy 2.x compatibility for TrackEval (older code uses deprecated aliases) ----
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.input_folder = config['paths']['input_folder']
        self.output_folder = config['paths']['output_folder']
        self.team_id = config['settings']['team_id']

        # Cartella temporanea necessaria per TrackEval
        self.tmp_root = os.path.abspath(os.path.join(self.output_folder, "_tmp_trackeval"))

    # =========================================================================
    # METODI HELPER
    # =========================================================================

    def _load_txt(self, path):
        """Legge file txt (serve solo per controlli rapidi o debug futuro)."""
        data = []
        if not os.path.exists(path): return np.empty((0, 6))
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
        data = {}
        if not os.path.exists(path): return data
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        data[(int(parts[0]), int(parts[1]))] = int(parts[2])
                    except ValueError:
                        continue
        return data

    def _generate_missing_behavior_gt(self, seq, gt_tracking_path):
        """Genera GT behavior mancante usando BBoxOperations."""
        gt_behavior_path = os.path.join(self.input_folder, seq, 'gt', 'behavior_gt.txt')
        if os.path.exists(gt_behavior_path): return gt_behavior_path

        roi_config_path = self.config['paths'].get('roi_config')
        if not roi_config_path or not os.path.exists(roi_config_path): return None

        with open(roi_config_path, 'r') as f:
            roi_data = json.load(f)
        IMG_W, IMG_H = 1920, 1080

        roi1_rect = BBoxOperations.get_absolute_roi(roi_data['roi1'], IMG_W, IMG_H)
        roi2_rect = BBoxOperations.get_absolute_roi(roi_data['roi2'], IMG_W, IMG_H)

        try:
            data = np.loadtxt(gt_tracking_path, delimiter=',')
        except Exception:
            return None

        if len(data) == 0: return None
        data = data.astype(float)
        unique_frames = np.unique(data[:, 0]).astype(int)
        lines_to_write = []

        for frame_idx in unique_frames:
            current_dets = data[data[:, 0] == frame_idx]
            c1, c2 = 0, 0
            for det in current_dets:
                x, y, w, h = det[2], det[3], det[4], det[5]
                feet_point = BBoxOperations.get_feet_point((x, y, w, h))
                if BBoxOperations.is_point_in_rect(roi1_rect, feet_point): c1 += 1
                if BBoxOperations.is_point_in_rect(roi2_rect, feet_point): c2 += 1
            lines_to_write.append(f"{frame_idx},1,{c1}\n")
            lines_to_write.append(f"{frame_idx},2,{c2}\n")

        try:
            os.makedirs(os.path.dirname(gt_behavior_path), exist_ok=True)
            with open(gt_behavior_path, 'w') as f:
                f.writelines(lines_to_write)
            return gt_behavior_path
        except Exception:
            return None

    def _compute_raw_mae_data(self, seq):
        gt_path = os.path.join(self.input_folder, seq, 'gt', 'gt.txt')
        gt_behavior_path = self._generate_missing_behavior_gt(seq, gt_path)
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
        return (10.0 - min(10.0, mae)) / 10.0

    # =========================================================================
    # EVALUATE PRINCIPALE
    # =========================================================================

    def evaluate(self, sequences=None):
        """
        Esegue la valutazione completa.
        Usa 'sequences' per filtrare l'output del report finale.
        """
        print("\n" + "=" * 125)
        print(f"{'SEQUENCE':<15} | {'HOTA':<7} | {'DetA':<7} | {'AssA':<7} | {'nMAE':<7} | {'TP':<6} | {'FN':<6} | {'FP':<6} |")
        print("-" * 125)

        # Preparazione Cartelle TrackEval
        try:
            gt_folder, tr_folder, seqmap_file = build_trackeval_structure(
                dataset_root=self.input_folder,
                predictions_root=self.output_folder,
                group=self.team_id,
                split="test",
                fps=25.0,
                tmp_root=self.tmp_root,
                benchmark="SNMOT",
                tracker_name="my_tracker",
                target_sequences=sequences
            )
        except Exception as e:
            print(f"Errore critico setup TrackEval: {e}")
            return

        # Esecuzione TrackEval (Silenziata)
        # TrackEval calcola su tutto quello che trova nelle cartelle
        official_results = compute_metrics_with_details(
            gt_folder=gt_folder,
            trackers_folder=tr_folder,
            seqmap_file=seqmap_file,
            split="test",
            benchmark="SNMOT",
            tracker_name="my_tracker"
        )

        final_rows = []
        global_raw_counts = {'TP': 0, 'FN': 0, 'FP': 0}

        # Definiamo il set di sequenze valide da mostrare (se specificato)
        valid_sequences = set(sequences) if (sequences and 'all' not in sequences) else None

        # Iterazione sui risultati
        for row in official_results:
            seq_name = row['Video']

            # FILTRO: Se l'utente ha chiesto sequenze specifiche, saltiamo quelle non richieste
            # (Manteniamo però GLOBAL_SCORE perché è il riassunto totale)
            if valid_sequences and seq_name != 'GLOBAL_SCORE' and seq_name not in valid_sequences:
                continue

            if seq_name == 'GLOBAL_SCORE':
                # --- GESTIONE RIGA GLOBALE ---
                beh_metrics = compute_nmae_from_behavior_files(
                    dataset_root=self.input_folder,
                    predictions_root=self.output_folder,
                    group=self.team_id,
                    sequences=valid_sequences
                )
                
                nmae = beh_metrics.get('nMAE', 0.0) or 0.0

                global_raw_counts['TP'] = row['TP']
                global_raw_counts['FN'] = row['FN']
                global_raw_counts['FP'] = row['FP']

            else:
                # --- GESTIONE SINGOLA SEQUENZA ---
                gt_path = os.path.join(self.input_folder, seq_name, 'gt', 'gt.txt')
                self._generate_missing_behavior_gt(seq_name, gt_path)

                abs_err, samples = self._compute_raw_mae_data(seq_name)
                local_mae = (abs_err / samples) if samples > 0 else 0.0
                nmae = self._normalize_mae(local_mae)

            # Salviamo il nMAE calcolato dentro il dizionario della riga
            row['nMAE'] = round(nmae, 6)
            final_rows.append(row)

            # Stampa riga tabella
            if seq_name == 'GLOBAL_SCORE': print("-" * 125)
            print(f"{seq_name:<15} | {row['HOTA'] * 100:6.4f}% | {row['DetA'] * 100:6.4f}% | {row['AssA'] * 100:6.4f}% | {nmae * 100:6.4f}% | {row['TP']:<6} | {row['FN']:<6} | {row['FP']:<6} |")

        print("-" * 125)

        # Estrazione Globale per PTBS e Salvataggio
        # Cerchiamo la riga globale nella lista filtrata
        global_row = next((r for r in final_rows if r['Video'] == 'GLOBAL_SCORE'), None)

        if global_row:
            # Calcolo PTBS finale usando i valori nel dizionario (senza variabili ridondanti)
            final_ptbs = global_row['HOTA'] + global_row['nMAE']
            
            print(f"{'PTBS':<15} | {final_ptbs:7.4f}")
            print("=" * 125 + "\n")

            self._save_results(
                results_list=[r for r in final_rows if r['Video'] != 'GLOBAL_SCORE'],
                avg_hota=global_row['HOTA'],
                global_nmae=global_row['nMAE'],
                avg_deta=global_row['DetA'],
                avg_assa=global_row['AssA'],
                final_ptbs=final_ptbs,
                global_raw_counts=global_raw_counts
            )

        if os.path.exists(self.tmp_root):
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _save_results(self, results_list, avg_hota, global_nmae, avg_deta, avg_assa, final_ptbs, global_raw_counts):
        """Salva JSON e appende al TXT (Config Tracker)."""

        # JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{self.team_id}_{timestamp}.json"
        save_path = os.path.join(self.output_folder, filename)

        tracker_cfg_path = self.config['paths']['tracker_config']
        if tracker_cfg_path and os.path.exists(tracker_cfg_path):
            tracker_cfg = yaml.safe_load(open(tracker_cfg_path, 'r'))
        
        structured_data = {
            "meta": {
                "timestamp": timestamp,
                "team_id": self.team_id
            },
            "main_config": self.config,
            "tracker_config": tracker_cfg,
            "metrics_overall": {
                "HOTA_05": round(float(avg_hota), 6),
                "nMAE": round(float(global_nmae), 6),
                "PTBS": round(float(final_ptbs), 6),
                "DetA": round(float(avg_deta), 6),
                "AssA": round(float(avg_assa), 6),
                "counts_sum": global_raw_counts
            },
            "metrics_per_sequence": results_list
        }
        try:
            with open(save_path, 'w') as f:
                json.dump(structured_data, f, indent=4)
            print(f"Report JSON salvato: {save_path}")
        except:
            pass

        # Append al Tracker Config per debug
        if tracker_cfg_path and os.path.exists(tracker_cfg_path):
            cfg_name = os.path.basename(tracker_cfg_path)
            saved_cfg_path = os.path.join(self.output_folder, cfg_name)
            if not os.path.exists(saved_cfg_path): shutil.copy(tracker_cfg_path, saved_cfg_path)

            try:
                with open(saved_cfg_path, 'a') as f:
                    f.write("\n\n# " + "=" * 129 + "\n")
                    f.write(f"# EVALUATION (TrackEval Official) - {datetime.now()}\n")
                    # Intestazione TXT senza MOTA
                    f.write(f"# {'SEQUENCE':<15} | {'HOTA':<7} | {'DetA':<7} | {'AssA':<7} | {'nMAE':<7} |\n")
                    f.write(f"# {'-' * 129}\n")
                    for res in results_list:
                        nm = res.get('nMAE', 0.0)
                        # Riga dati senza MOTA
                        f.write(
                            f"# {res['Video']:<15} | {res['HOTA'] * 100:6.2f}% | {res['DetA'] * 100:6.2f}% | {res['AssA'] * 100:6.2f}% | {nm * 100:6.2f}% |\n")
                    f.write(f"# {'-' * 129}\n")
                    # Riga globale senza MOTA
                    f.write("# " + "-" * 130)
                    f.write(f"# {'GLOBAL':<15} | {avg_hota * 100:6.2f}% | {avg_deta * 100:6.2f}% | {avg_assa * 100:6.2f}% | {global_nmae * 100:6.2f}% |\n")
                    f.write("# " + "-" * 130)
                    f.write(f"# {'PTBS':<20} | {final_ptbs:7.4f}\n")
                    f.write("# " + "=" * 130 + "\n")
                print(f"Report TXT aggiunto a: {saved_cfg_path}")
            except Exception as e:
                print(f"Errore scrittura TXT: {e}")