import os
import torch
import gc
import time
import shutil
import json
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from utils.field_masking import *
from utils.bbox_operations import BBoxOperations
from utils.bbox_drawer import BBoxDrawer

class SoccerTracker:
    def __init__(self, config):
        """Inizializza il tracker leggendo i parametri dal config."""
        self.config = config
        self.paths = config['paths']
        self.sets = config['tracker']
        self.team_id = config['settings']['team_id']

        print(f"Detection model: {self.paths['detection_model']} (classes: {self.sets['classes']})")
        self.model = YOLO(self.paths['detection_model'])

        self.drawer = BBoxDrawer()
        self.tracker_cfg_path = self.paths['tracker_config']
        tracker_cfg = os.path.basename(self.tracker_cfg_path)
        tracker_cfg = yaml.safe_load(open(self.tracker_cfg_path, 'r'))
        self.with_reid = tracker_cfg.get('with_reid', False)
        self.output_folder = self.paths['output_folder']

        self.imgsz = self.sets.get('imgsz', 1088)
        self.conf = self.sets.get('conf', 0.2)
        self.iou = self.sets.get('iou', 0.7)
        self.batch_size = self.sets.get('batch', 16)
        self.device = self.sets.get('device', 0)
        self.classes = self.sets.get('classes', [0])
        self.verbose = self.sets.get('verbose', False)
        self.half = self.sets.get('half', False)

        self.use_field_mask = self.sets.get('use_field_mask', True)
        self.mask_frequency = self.sets.get('mask_frequency', 1)
        self.buffer_size = self.sets.get('buffer_size', 250)

        # Flag Debug
        self.debug = False
        self.show_behaviour = False
        self.show_mask_overlay = False
        self.show_track = False

        debug_cfg = config.get('debug', False)

        if debug_cfg is True or isinstance(debug_cfg, dict):
            self.debug = True
            # Su Colab forziamo batch=1 per permettere la visualizzazione frame-by-frame
            self.batch_size = 1

            if isinstance(debug_cfg, dict):
                self.show_mask_overlay = debug_cfg.get('show_mask', False)
                self.show_behaviour = debug_cfg.get('show_behaviour', False)
                self.show_track = debug_cfg.get('show_track', False)
            else:
                self.show_mask_overlay = True
                self.show_behaviour = True
                self.show_track = True

        # Caricamento ROI (solo se debug è attivo)
        self.roi_data = {}
        if self.debug:
            try:
                roi_path = self.paths['roi_config']
                with open(roi_path, 'r') as f:
                    self.roi_data = json.load(f)
            except Exception:
                print(f"ROI Config non trovato o errore: {self.paths['roi_config']}")
        
        print(f"\nTracker creato = imgsz: {self.imgsz} | conf: {self.conf} | iou: {self.iou} | batch: {self.batch_size} | reid: {self.with_reid} | device: {self.device} | half-precision (FP16): {self.half} | verbose: {self.verbose} | debug: (track: {self.show_track}, behav: {self.show_behaviour})")

    def track_sequence(self, sequence_name):
        """Esegue il tracking su una sequenza specifica."""

        # 1. Configurazione Percorsi
        source_path = os.path.join(self.paths['input_folder'], sequence_name, "img1")
        output_dir = self.paths['output_folder']
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"tracking_{sequence_name}_{self.team_id}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n- Avvio Tracking su sequenza {sequence_name} |")
        # 2. Pulizia Memoria
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Parametri YOLO
        track_params = {
            'source': source_path,
            'tracker': self.tracker_cfg_path,
            'imgsz': self.imgsz,
            'conf': self.conf,
            'iou': self.iou,
            'batch': self.batch_size,
            'device': self.device,
            'classes': self.classes,
            'verbose': self.verbose,
            'persist': True,
            'stream': True,
            'show': False # Importante: disabilita show nativo di YOLO
        }

        results = self.model.track(**track_params)

        # 4. Inizializzazione Ciclo
        field_mask = None
        start_time = time.time()

        with open(output_path, 'w') as f:
            buffer = []

            # 5. Loop sui Frame
            for frame_idx, r in enumerate(results):
                img = r.orig_img
                frame_log_id = frame_idx + 1

                # A. Gestione Maschera Campo
                if self.use_field_mask:
                    # Calcola maschera se: frequenza raggiunta OR prima volta OR serve per debug visivo
                    if frame_idx % self.mask_frequency == 0 or field_mask is None or self.show_mask_overlay:
                        field_mask = get_field_mask(img)
                else:
                    if field_mask is None:
                        field_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

                det_to_draw = []

                # B. Estrazione Detection
                if r.boxes is not None and r.boxes.id is not None:
                    boxes_xywh = r.boxes.xywh.cpu().numpy()
                    track_ids = r.boxes.id.cpu().numpy()

                    for box, t_id in zip(boxes_xywh, track_ids):
                        tl_x, tl_y, w, h = BBoxOperations.center_to_top_left(box)
                        feet_point = BBoxOperations.get_feet_point((tl_x, tl_y, w, h))

                        # Filtro Logico
                        if not self.use_field_mask or is_point_on_field(feet_point, field_mask, bottom_tolerance=40):
                            buffer.append(f"{frame_log_id},{int(t_id)},{tl_x},{tl_y},{w},{h}\n")
                            if self.debug:
                                det_to_draw.append((tl_x, tl_y, w, h, int(t_id)))

                # C. Visualizzazione Debug (Safety Check)
                if self.debug:
                    # Se la visualizzazione fallisce, disattiviamo il debug per evitare crash continui
                    if not self._drawer_debug(img, det_to_draw, field_mask, sequence_name, frame_idx):
                        print("Errore visualizzazione: disattivo debug grafico.")
                        self.debug = False

                # D. Scrittura su Disco
                if frame_log_id % self.buffer_size == 0:
                    f.writelines(buffer)
                    buffer.clear()
                    f.flush()

            # 6. Flush Finale
            if buffer:
                f.writelines(buffer)
                f.flush()

        # 7. Chiusura Debug
        if self.debug:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass

        # 8. Conclusione
        elapsed_time = time.time() - start_time
        print(f"Tracking completato in {elapsed_time:.2f} secondi.")

        self._copy_tracker_config()
        torch.cuda.empty_cache()

        return output_path

    def _drawer_debug(self, img, det_to_draw, field_mask, sequence_name, frame_idx):
        """
        Disegna il frame di debug.
        Supporta sia Colab (cv2_imshow) che Locale (cv2.imshow).
        """
        try:
            # Lavora su una copia per non sporcare l'originale
            canvas = img.copy()
            h_img, w_img = canvas.shape[:2]

            # 1. Maschera Campo
            if self.show_mask_overlay and field_mask is not None:
                green_overlay = np.zeros_like(canvas)
                green_overlay[field_mask > 0] = [0, 255, 0]
                cv2.addWeighted(canvas, 1.0, green_overlay, 0.3, 0, canvas) # 0.3 = Opacità maschera
                contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

            # 2. Behaviour (ROI)
            if self.show_behaviour:
                for key in ['roi1', 'roi2']:
                    if key in self.roi_data:
                        roi_rect = BBoxOperations.get_absolute_roi(self.roi_data[key], w_img, h_img)
                        count = 0
                        for (x, y, w, h, _) in det_to_draw:
                            feet_point = BBoxOperations.get_feet_point((x, y, w, h))
                            if BBoxOperations.is_point_in_rect(roi_rect, feet_point):
                                count += 1
                        self.drawer.draw_roi(canvas, roi_rect, f"{key.upper()}: {count}", key)

            # 3. Tracking (Box)
            if self.show_track:
                for (x, y, w, h, t_id) in det_to_draw:
                    self.drawer.draw_player(canvas, (x, y, w, h), t_id)

            # 4. Info
            info_text = f"LIVE: {sequence_name} | Frame: {frame_idx}"
            cv2.putText(canvas, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow(f"Tracker Debug", canvas)
            cv2.waitKey(1)

            return True

        except Exception as e:
            return False

    def _copy_tracker_config(self):
        if self.tracker_cfg_path and os.path.exists(str(self.tracker_cfg_path)):
            try:
                src_path = str(self.tracker_cfg_path)
                cfg_filename = os.path.basename(src_path)
                dst_path = os.path.join(self.paths['output_folder'], cfg_filename)
                shutil.copy(src_path, str(dst_path))
            except Exception:
                pass