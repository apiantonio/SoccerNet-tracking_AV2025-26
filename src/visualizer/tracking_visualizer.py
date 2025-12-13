import os
import glob
import cv2
import numpy as np
from tracker_config import TrackerConfig

class TrackingVisualizer:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self.fps = 25
        self.colors = {}

    def _get_color(self, id):
        if id not in self.colors:
            np.random.seed(id)
            self.colors[id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[id]

    def _load_tracking_data(self, filepath):
        if not os.path.exists(filepath):
            print(f"❌ File dati non trovato: {filepath}")
            return {}

        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                
                fid = int(parts[0])
                tid = int(parts[1])
                x = int(float(parts[2]))
                y = int(float(parts[3]))
                w = int(float(parts[4]))
                h = int(float(parts[5]))
                
                if fid not in data: data[fid] = []
                data[fid].append({'id': tid, 'bbox': (x, y, x+w, y+h)})
        return data

    def create_video(self, sequence_name, tracking_file=None):
        """
        Genera il video annotato per una sequenza specifica.
        Se tracking_file è None, cerca quello automatico generato dal Tracker.
        """
        # Percorsi input
        seq_path = os.path.join(self.cfg.dataset_dir, sequence_name)
        img_folder = os.path.join(seq_path, 'img1')
        
        if tracking_file is None:
             tracking_file = os.path.join(self.cfg.output_dir, f"tracking_{sequence_name}_{self.cfg.team_id}.txt")

        # Percorso output video
        output_video = os.path.join(self.cfg.output_dir, f"vis_{sequence_name}.mp4")

        # Caricamento dati
        tracks = self._load_tracking_data(tracking_file)
        if not tracks:
            print(f"⚠️ Nessun dato di tracking trovato per {sequence_name}. Salto visualizzazione.")
            return

        img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
        if not img_files:
            print("❌ Immagini non trovate.")
            return

        # Setup Video Writer
        first = cv2.imread(img_files[0])
        h, w, _ = first.shape
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))

        print(f"🎨 Generazione video: {output_video}")
        
        limit = self.cfg.limit_frames if self.cfg.limit_frames else len(img_files)
        
        for i, path in enumerate(img_files):
            if i >= limit: break
            
            frame = cv2.imread(path)
            frame_id = i + 1
            
            if i % 100 == 0: print(f"Rendering {i}/{limit}...")

            if frame_id in tracks:
                for obj in tracks[frame_id]:
                    x1, y1, x2, y2 = obj['bbox']
                    tid = obj['id']
                    color = self._get_color(tid)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    text = f"ID {tid}"
                    (tw, th), _ = cv2.getTextSize(text, 0, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1-15), (x1+tw, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            out.write(frame)
        
        out.release()
        print("✅ Video salvato.")