import os
import cv2
import json
import numpy as np
import glob

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

    def _get_color(self, id):
        """Genera un colore univoco per ogni ID"""
        if id not in self.colors:
            np.random.seed(int(id))
            self.colors[id] = tuple(np.random.randint(0, 255, 3).tolist())
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

    def generate_video(self, sequence_name):
        """Genera il video annotato per la sequenza data"""
        
        # Costruzione percorsi
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
        fps = 25 # Standard per SoccerNet
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_img, h_img))

        # Recupero ROI per questa sequenza
        roi1_rect = self._get_absolute_roi(self.roi_data['roi1'], w_img, h_img)
        roi2_rect = self._get_absolute_roi(self.roi_data['roi2'], w_img, h_img)

        print(f"🎬 Rendering Video: {sequence_name} -> {output_video_path}")

        for i, img_path in enumerate(img_files):
            frame = cv2.imread(img_path)
            frame_id = i + 1 # I file partono da 1 solitamente nel tracking

            # 1. Disegna le ROI (Sfondo semi-trasparente opzionale o solo bordo)
            if roi1_rect:
                rx, ry, rw, rh = roi1_rect
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3) # Rosso per ROI 1
                cv2.putText(frame, "ROI 1", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            if roi2_rect:
                rx, ry, rw, rh = roi2_rect
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 3) # Blu per ROI 2
                cv2.putText(frame, "ROI 2", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            # 2. Disegna i Tracking
            if frame_id in tracks:
                for obj in tracks[frame_id]:
                    x, y, w, h = obj['bbox']
                    tid = obj['id']
                    color = self._get_color(tid)
                    
                    # Box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    # Label ID
                    cv2.putText(frame, f"{tid}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. Overlay Statistiche Behaviour (In alto a sinistra)
            if frame_id in behaviors:
                b_data = behaviors[frame_id]
                count1 = b_data.get(1, 0) # Conteggio ROI 1
                count2 = b_data.get(2, 0) # Conteggio ROI 2
                
                # Pannello info
                cv2.rectangle(frame, (0, 0), (350, 100), (0,0,0), -1) # Sfondo nero
                cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"ROI 1 (Red):  {count1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(frame, f"ROI 2 (Blue): {count2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            out.write(frame)
            if i % 100 == 0:
                print(f"   Processed frame {i}/{len(img_files)}")

        out.release()
        print(f"✅ Video completato: {output_video_path}")