import os
import torch
import gc
import time 
import shutil
import yaml
import cv2
import json  # Aggiunto per leggere config ROI
import numpy as np
from ultralytics import YOLO
from utils.field_masking import *

class SoccerTracker:
    def __init__(self, config):
        """
        Inizializza il tracker leggendo i parametri dal config.
        config: dizionario caricato da main_config.yaml
        """
        self.config = config
        
        self.model_path = config['paths']['detection_model']
        self.tracker_cfg_path = config['paths']['tracker_config']
        self.conf = config['tracker']['conf']
        self.iou = config['tracker']['iou']
        self.batch_size = config['tracker']['batch']
        self.use_field_mask = config['tracker'].get('field_mask', True)

        self.device = config['tracker']['device']
        self.output_folder = config['paths']['output_folder']
        self.classes = config['tracker'].get('classes', [0])
        self.verbose = config['tracker'].get('verbose', False)
        
        # Parametri per ottimizzazione memoria
        self.imgsz = config['tracker'].get('imgsz', 1088)
        self.half = config['tracker'].get('half', False)
        
        # Variabili di supporto
        self._mask_frequency = config['tracker'].get('mask_frequency', 1)  # Numero di frame tra un aggiornamento della maschera
        self._buffer_size = config['tracker'].get('buffer_size', 250)      # Numero di frame dopo i quali scrivere su file
        
        # Per debug mode
        debug_cfg = config.get('debug', False)
        
        # Inizializza flag a False
        self.debug = False
        self.show_track = False
        self.show_behaviour = False
        self.show_mask_overlay = False
       
        if isinstance(debug_cfg, dict):
            # Se arriva dal main.py processato (è un dizionario)
            self.debug = True
            self.show_track = debug_cfg.get('show_track', False)
            self.show_behaviour = debug_cfg.get('show_behaviour', False)
            self.show_mask_overlay = debug_cfg.get('show_mask', False)
            self.batch_size = 1 # altrimenti non funziona bene la visualizzazione
            self.color_cache = {} 
        elif debug_cfg is True:
            # Se attivato manualmente nel yaml come 'True' generico
            self.debug = True
            self.show_track = True
            self.show_behaviour = True
            self.show_mask_overlay = True
            self.batch_size = 1
            self.color_cache = {}
        
        # Per visualizzazione debug
        self.roi_path = config['paths']['roi_config']
        self.roi_data = {}
        if os.path.exists(self.roi_path):
            with open(self.roi_path, 'r') as f:
                self.roi_data = json.load(f)
        else:
            print(f"⚠️ ROI Config non trovato: {self.roi_path}")

        # Palette colori
        self.COLOR_ROI1 = (0, 0, 255)       # Rosso
        self.COLOR_ROI2 = (255, 0, 0)       # Blu
        self.COLOR_TEXT_BG = (40, 40, 40)   # Grigio scuro
        self.COLOR_TEXT = (255, 255, 255)   # Bianco
    
        print(f"🔄 Caricamento Modello YOLO: {self.model_path}")
        self.model = YOLO(self.model_path)
        
    def _get_id_color(self, track_id):
        """Genera un colore univoco e consistente per ogni track_id."""
        track_id = int(track_id)
        if track_id not in self.color_cache:
            np.random.seed(track_id)
            color = np.random.randint(50, 255, size=3).tolist()
            self.color_cache[track_id] = tuple(color)
        return self.color_cache[track_id]

    # --- METODI HELPER PORTATI DAL VISUALIZER TODO Refactoring ---
    def _get_absolute_roi(self, roi_relative, img_w, img_h):
        x = int(roi_relative['x'] * img_w)
        y = int(roi_relative['y'] * img_h)
        w = int(roi_relative['width'] * img_w)
        h = int(roi_relative['height'] * img_h)
        return x, y, w, h

    def _draw_transparent_box(self, img, pt1, pt2, color, alpha=0.4):
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_stylish_tag(self, img, text, center_x, top_y, bg_color=(0,0,0)):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x1 = int(center_x - text_w // 2 - 4)
        y1 = int(top_y)
        x2 = int(center_x + text_w // 2 + 4)
        y2 = int(top_y + text_h + 10)
        
        self._draw_transparent_box(img, (x1, y1), (x2, y2), bg_color, alpha=0.6)
        
        text_x = int(center_x - text_w // 2)
        text_y = int(top_y + text_h + 5)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, self.COLOR_TEXT, thickness, cv2.LINE_AA)
    # --------------------------------------------

    def track_sequence(self, sequence_name):
        """
        Esegue il tracking. Se show_video=True usa lo stile del Visualizer.
        """

        source_path = os.path.join(self.config['paths']['input_folder'], sequence_name, "img1")
        output_dir = self.config['paths']['output_folder']
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"tracking_{sequence_name}_{self.config['settings']['team_id']}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        tracker_cfg = os.path.basename(self.tracker_cfg_path)
        tracker_cfg = yaml.safe_load(open(self.tracker_cfg_path, 'r'))
        if tracker_cfg.get('with_reid', False):
            print(f"\n🔍 ReID abilitato.")
        else:
            print("\n🔍 ReID non abilitato.")

        print(f"🚀 Avvio Tracking: {sequence_name} | imgsz: {self.imgsz} | conf: {self.conf} | iou: {self.iou} | batch: {self.batch_size} | device: {self.device} | half-precision (FP16): {self.half} | verbose: {self.verbose} | debug: (track: {self.show_track}, behav: {self.show_behaviour})")

        gc.collect()
        torch.cuda.empty_cache()

        results = self.model.track(
            source=source_path,
            tracker=self.tracker_cfg_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            batch=self.batch_size,
            half=self.half,
            device=self.device,
            classes=self.classes,
            verbose=self.verbose,
            persist=True,
            stream=True, 
            show=False 
        )
        
        window_name = f"Tracking Debug (Visualizer Style) - {sequence_name}"
        if self.debug:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1920, 1080)
    
        tracking_start_time = time.time()
        
        try:
            with open(output_path, 'w') as f:
                buffer = [] # Buffer per scrittura file
                
                for frame_idx, r in enumerate(results):
                    
                    if self.use_field_mask: # FIELD MASKING & FILTERING:
                        # Genera la maschera del campo per il frame corrente
                        # NOTA: Se è troppo lento, puoi considerare di ridimensionare 
                        # l'immagine prima di calcolare la maschera o calcolarla ogni N frame.
                        if frame_idx % self._mask_frequency == 0:  # Aggiorna la maschera ogni N frame
                            field_mask = get_field_mask(r.orig_img)
                    else:
                        # Maschera piena (tutto il frame)
                        h_img, w_img = r.orig_img.shape[:2]
                        field_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
                    
                    # Liste per contenere solo i dati validi (dentro il campo)
                    valid_boxes_xywh = []
                    valid_boxes_xyxy = []
                    valid_ids = []
                    valid_confs = []

                    # Controlla se YOLO ha trovato qualcosa
                    if r.boxes is not None and r.boxes.id is not None:
                        # Estrai i dati in numpy per iterazione veloce
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        boxes_xywh = r.boxes.xywh.cpu().numpy()
                        track_ids = r.boxes.id.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()

                        for i, box in enumerate(boxes_xyxy):
                            x1, y1, x2, y2 = box
                            
                            # Calcola il punto dei piedi (base centrale del box)
                            feet_point = (int((x1 + x2) / 2), int(y2))
                            
                            # Verifica se è nel campo usando la funzione di utils
                            if is_point_on_field(feet_point, field_mask, bottom_tolerance=40):
                                valid_boxes_xywh.append(boxes_xywh[i])
                                valid_boxes_xyxy.append(box)
                                valid_ids.append(track_ids[i])
                                valid_confs.append(confs[i])
                            else:
                                # DEBUG: Punto del bounding box ignorato con z molto alta
                                cv2.drawMarker(r.orig_img, feet_point, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
                                pass
                    
                    # --- LOGICA VISUALIZZAZIONE "DEBUG MODE" ---
                    if self.debug:
                        frame_display = r.orig_img.copy()
                        h_img, w_img = frame_display.shape[:2]
                        
                        # Visualizza la maschera in overlay verdino per debug
                        if self.show_mask_overlay:
                            # Crea overlay verde dove la maschera è attiva
                            green_overlay = np.zeros_like(frame_display)
                            green_overlay[field_mask > 0] = [0, 255, 0] # Canale G
                            cv2.addWeighted(frame_display, 1.0, green_overlay, 0.2, 0, frame_display)
                            # Disegna contorno campo
                            contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame_display, contours, -1, (0, 255, 0), 2)
                        
                        # Visualizza ROI e conteggi comportamento
                        if self.show_behaviour:
                            # 1. Recupero coord. assolute ROI
                            roi1_rect = self._get_absolute_roi(self.roi_data.get('roi1'), w_img, h_img) if 'roi1' in self.roi_data else None
                            roi2_rect = self._get_absolute_roi(self.roi_data.get('roi2'), w_img, h_img) if 'roi2' in self.roi_data else None

                            # 2. Calcolo conteggi LIVE per le ROI (Passaggio preliminare)
                            count1, count2 = 0, 0
                            for i, box in enumerate(valid_boxes_xywh):
                                bx, by, bw, bh = box
                                feet_x = int(bx)
                                feet_y = int(by + bh / 2)
                                
                                if roi1_rect:
                                    rx, ry, rw, rh = roi1_rect
                                    if rx <= feet_x <= rx + rw and ry <= feet_y <= ry + rh:
                                        count1 += 1
                                if roi2_rect:
                                    rx, ry, rw, rh = roi2_rect
                                    if rx <= feet_x <= rx + rw and ry <= feet_y <= ry + rh:
                                        count2 += 1

                            # 3. Disegno Sfondi ROI e Header
                            if roi1_rect:
                                rx, ry, rw, rh = roi1_rect
                                self._draw_transparent_box(frame_display, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI1, alpha=0.15)
                                cv2.rectangle(frame_display, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI1, 2)
                                self._draw_stylish_tag(frame_display, f"ROI 1 | Players: {count1}", rx + rw//2, ry - 25, bg_color=self.COLOR_ROI1)

                            if roi2_rect:
                                rx, ry, rw, rh = roi2_rect
                                self._draw_transparent_box(frame_display, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI2, alpha=0.15)
                                cv2.rectangle(frame_display, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI2, 2)
                                self._draw_stylish_tag(frame_display, f"ROI 2 | Players: {count2}", rx + rw//2, ry - 25, bg_color=self.COLOR_ROI2)

                        # visualizza i box tracciati
                        if self.show_track:
                            # Disegna solo i giocatori VALIDI (in campo)
                            for i, box in enumerate(valid_boxes_xyxy):
                                x1, y1, x2, y2 = map(int, box)
                                tid = int(valid_ids[i])
                                conf = valid_confs[i]
                                color = self._get_id_color(tid)
                                
                                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                                feet_x = int((x1 + x2) / 2)
                                feet_y = int(y2)
                                cv2.circle(frame_display, (feet_x, feet_y), 4, color, -1)

                                # Tag ID | Conf
                                label_text = f"ID {tid} | {conf:.2f}"
                                self._draw_stylish_tag(frame_display, label_text, feet_x, feet_y + 8, bg_color=self.COLOR_TEXT_BG)

                        # Info Frame in basso
                        info_text = f"Frame: {frame_idx + 1} | On Field: {len(valid_ids)} | Ignored: {len(r.boxes) - len(valid_ids) if r.boxes else 0}"
                        cv2.putText(frame_display, info_text, (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

                        cv2.imshow(window_name, frame_display)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print(f"⏩ Skip sequence {sequence_name}...")
                            break

                    if r.boxes is None or r.boxes.id is None:
                        continue
                    
                    # Scrittura su file (bufferizzata)
                    for i, box in enumerate(valid_boxes_xywh):
                        track_id = valid_ids[i]
                        x, y, w, h = box
                        # Coordinate Top-Left come richiesto dal formato output
                        x1 = x - (w / 2)
                        y1 = y - (h / 2)
                        # Scrivi nel buffer
                        buffer.append(f"{frame_idx + 1},{int(track_id)},{int(x1)},{int(y1)},{int(w)},{int(h)}\n")
                    
                    # Scrivi il buffer solo alla fine per evitare I/O frequenti
                    if (frame_idx + 1) % self._buffer_size == 0:
                        f.writelines(buffer)
                        buffer.clear()
                        f.flush()
        
        finally:
            if self.debug:
                try:
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass
        
        elapsed_time = time.time() - tracking_start_time
        print(f"⏱️ Tracking e scrittura completato in {elapsed_time:.2f} secondi.")
        
        self._copy_tracker_config()
        torch.cuda.empty_cache()
        return output_path
    
    def _copy_tracker_config(self):
        if os.path.exists(self.tracker_cfg_path):
            cfg_filename = os.path.basename(self.tracker_cfg_path)
            self.dst_cfg_path = os.path.join(self.output_folder, cfg_filename)
            try:
                shutil.copy(self.tracker_cfg_path, self.dst_cfg_path)
            except Exception as e:
                print(f"⚠️ Impossibile copiare il config del tracker: {e}")