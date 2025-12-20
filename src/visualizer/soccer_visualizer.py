import os
import cv2
import json
import numpy as np
import glob
from utils.field_masking import get_field_mask, is_point_on_field # Importata anche la funzione di verifica

class SoccerVisualizer:
    def __init__(self, config):
        """
        Inizializza il visualizzatore.
        config: dizionario caricato da yaml
        """
        self.config = config
        self.colors = {}  # Cache per i colori degli ID
        self.roi_path = config['paths']['roi_config']
        
        # Carica configurazione ROI
        with open(self.roi_path, 'r') as f:
            self.roi_data = json.load(f)

        # Palette colori stile TV
        self.COLOR_ROI1 = (0, 0, 255)       # Rosso
        self.COLOR_ROI2 = (255, 0, 0)       # Blu
        self.COLOR_TEXT_BG = (40, 40, 40)   # Grigio scuro per background label
        self.COLOR_TEXT = (255, 255, 255)   # Bianco

    def _get_color(self, id):
        """Genera un colore univoco per ogni ID"""
        if id not in self.colors:
            np.random.seed(int(id))
            # Colori pastello/vivaci per i box, evitando colori troppo scuri
            self.colors[id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.colors[id]

    def _load_tracking_data(self, filepath):
        """Carica il file tracking_K_XX.txt"""
        data = {}
        if not os.path.exists(filepath):
            print(f"⚠️ File tracking non trovato: {filepath}")
            return data

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                # Format: frame, id, x, y, w, h
                fid = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                
                if fid not in data: data[fid] = []
                data[fid].append({'id': tid, 'bbox': (int(x), int(y), int(w), int(h))})
        return data

    def _load_behavior_data(self, filepath):
        """Carica il file behavior_K_XX.txt"""
        # Format: frame, region_id, count
        data = {}
        if not os.path.exists(filepath):
            print(f"⚠️ File behaviour non trovato: {filepath}")
            return data

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3: continue
                
                fid = int(parts[0])
                rid = int(parts[1])
                count = int(parts[2])
                
                if fid not in data: data[fid] = {}
                data[fid][rid] = count
        return data

    def _get_absolute_roi(self, roi_relative, img_w, img_h):
        """Converte coordinate relative (0-1) in pixel"""
        x = int(roi_relative['x'] * img_w)
        y = int(roi_relative['y'] * img_h)
        w = int(roi_relative['width'] * img_w)
        h = int(roi_relative['height'] * img_h)
        return x, y, w, h

    def _draw_transparent_box(self, img, pt1, pt2, color, alpha=0.4):
        """Disegna un rettangolo con riempimento trasparente"""
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_stylish_tag(self, img, text, center_x, top_y, bg_color=(0,0,0)):
        """Disegna una label centrata con sfondo trasparente sotto i piedi"""
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Coordinate del box di sfondo
        x1 = int(center_x - text_w // 2 - 4)
        y1 = int(top_y)
        x2 = int(center_x + text_w // 2 + 4)
        y2 = int(top_y + text_h + 10)
        
        # Disegna sfondo trasparente
        self._draw_transparent_box(img, (x1, y1), (x2, y2), bg_color, alpha=0.6)
        
        # Disegna testo
        text_x = int(center_x - text_w // 2)
        text_y = int(top_y + text_h + 5)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, self.COLOR_TEXT, thickness, cv2.LINE_AA)

    def generate_video(self, sequence_name):
        """Genera il video annotato per la sequenza data"""
        
        # Costruzione percorsi (LOGICA ORIGINALE)
        img_folder = os.path.join(self.config['paths']['input_folder'], sequence_name, "img1")
        output_dir = self.config['paths']['output_folder']
        team_id = self.config['settings']['team_id']
        
        track_file = os.path.join(output_dir, f"tracking_{sequence_name}_{team_id}.txt")
        behav_file = os.path.join(output_dir, f"behavior_{sequence_name}_{team_id}.txt")
        output_video_path = os.path.join(output_dir, f"video_{sequence_name}_{team_id}.mp4")
        
        # Caricamento Dati
        tracks = self._load_tracking_data(track_file)
        behaviors = self._load_behavior_data(behav_file)
        
        # Recupero Immagini
        img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
        if not img_files:
            print(f"❌ Nessuna immagine trovata in {img_folder}")
            return

        # Setup Video Writer
        first_frame = cv2.imread(img_files[0])
        h_img, w_img, _ = first_frame.shape
        fps = 25 
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_img, h_img))

        # Recupero ROI per questa sequenza
        roi1_rect = self._get_absolute_roi(self.roi_data['roi1'], w_img, h_img) if 'roi1' in self.roi_data else None
        roi2_rect = self._get_absolute_roi(self.roi_data['roi2'], w_img, h_img) if 'roi2' in self.roi_data else None

        print(f"🎬 Rendering Video: {sequence_name} -> {output_video_path}")

        for i, img_path in enumerate(img_files):
            frame = cv2.imread(img_path)
            frame_id = i + 1 
            
            # Calcola la maschera per il frame corrente
            field_mask = get_field_mask(frame)
            # Estrai i contorni dalla maschera
            contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Disegna il contorno in Verde Lime (BGR: 0, 255, 0)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            # Recupera conteggi per il frame corrente
            count1 = 0
            count2 = 0
            if frame_id in behaviors:
                count1 = behaviors[frame_id].get(1, 0)
                count2 = behaviors[frame_id].get(2, 0)

            # Disegna ROI 1 (Rosso)
            if roi1_rect:
                rx, ry, rw, rh = roi1_rect
                # Riempimento molto leggero
                self._draw_transparent_box(frame, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI1, alpha=0.15)
                # Bordo più solido
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI1, 2)
                # Header ROI con conteggio
                header_text = f"ROI 1 | Players: {count1}"
                self._draw_stylish_tag(frame, header_text, rx + rw//2, ry - 25, bg_color=self.COLOR_ROI1)

            # Disegna ROI 2 (Blu)
            if roi2_rect:
                rx, ry, rw, rh = roi2_rect
                self._draw_transparent_box(frame, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI2, alpha=0.15)
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), self.COLOR_ROI2, 2)
                header_text = f"ROI 2 | Players: {count2}"
                self._draw_stylish_tag(frame, header_text, rx + rw//2, ry - 25, bg_color=self.COLOR_ROI2)

            # --- 2. TRACKING E ID GIOCATORI ---
            if frame_id in tracks:
                for obj in tracks[frame_id]:
                    x, y, w, h = obj['bbox']
                    tid = obj['id']
                    
                    # Calcolo centro della base (piedi)
                    feet_x = int(x + w / 2)
                    feet_y = int(y + h)
                    
                    # --- NUOVA LOGICA: Controllo Maschera ---
                    # Verifica se il giocatore è "in campo" secondo la maschera corrente.
                    # Nota: Uso bottom_tolerance=40 per coerenza con il tracker
                    if is_point_on_field((feet_x, feet_y), field_mask, bottom_tolerance=40):
                        # CASO 1: Giocatore VALID (In campo) -> Disegna Box + ID
                        color = self._get_color(tid)

                        # Bounding Box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # "Mirino" sui piedi
                        cv2.circle(frame, (feet_x, feet_y), 4, color, -1)
                        cv2.circle(frame, (feet_x, feet_y), 5, (0,0,0), 1)

                        # Label ID
                        self._draw_stylish_tag(frame, f"ID {tid}", feet_x, feet_y + 8, bg_color=self.COLOR_TEXT_BG)
                    
                    else:
                        # CASO 2: Giocatore SCARTATO (Fuori campo) -> Disegna Croce Rossa
                        # Come nel soccer_tracker: markerType=cv2.MARKER_TILTED_CROSS
                        cv2.drawMarker(frame, (feet_x, feet_y), (0, 0, 255), 
                                     markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2)
                        # Opzionale: Se vuoi vedere anche il box "scartato", puoi decommentare sotto:
                        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

            # --- 3. INFO GENERALI ---
            info_text = f"Frame: {frame_id} | Total tracked: {len(tracks.get(frame_id, []))}"
            cv2.putText(frame, info_text, (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 1)

            out.write(frame)
            if i % 100 == 0:
                print(f"   Processed frame {i}/{len(img_files)}")

        out.release()
        print(f"✅ Video completato: {output_video_path}")