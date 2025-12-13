import os
import glob
from ultralytics import YOLO
from src.tracker_config import TrackerConfig


class SoccerTracker:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        print(f"🔄 Caricamento modello da: {self.cfg.model_path}")
        self.model = YOLO(self.cfg.model_path)

    def run_on_sequence(self, sequence_name):
        """
        Esegue il tracking su una specifica cartella (es. SNMOT-060)
        """
        sequence_path = os.path.join(self.cfg.dataset_dir, sequence_name)
        img_dir = os.path.join(sequence_path, 'img1')
        
        if not os.path.exists(img_dir):
            print(f"⚠️ Cartella immagini non trovata: {img_dir}")
            return None

        # Nome file output conforme al PDF (tracking_K_XX.txt)
        # Qui usiamo il nome sequenza, rinominare se necessario k=1,2,3
        output_filename = f"tracking_{sequence_name}_{self.cfg.team_id}.txt"
        output_path = os.path.join(self.cfg.output_dir, output_filename)

        print(f"🚀 Avvio Tracking su: {sequence_name}")
        print(f"💾 Output sarà salvato in: {output_path}")

        results = self.model.track(
            source=img_dir,
            persist=True,
            tracker=self.cfg.tracker_config,
            conf=self.cfg.conf_thresh,
            imgsz=self.cfg.img_size,
            classes=self.cfg.classes,
            verbose=False,
            device=0 # Cambia in 'cpu' se non hai GPU
        )

        self._write_results(results, output_path)
        return output_path

    def _write_results(self, results, output_path):
        with open(output_path, 'w') as f:
            for frame_idx, r in enumerate(results):
                
                # Check limite frame per test
                if self.cfg.limit_frames and frame_idx >= self.cfg.limit_frames:
                    print(f"🛑 Stop manuale al frame {frame_idx}")
                    break

                frame_id = frame_idx + 1 # MOT format start from 1

                if r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.int().cpu().numpy()
                    
                    for box, track_id in zip(boxes, ids):
                        x1, y1, x2, y2 = box
                        
                        # Conversione in formato MOT (top_left_x, top_left_y, w, h)
                        tl_x = int(x1)
                        tl_y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        
                        # Scrittura: frame, id, x, y, w, h
                        line = f"{frame_id},{track_id},{tl_x},{tl_y},{w},{h}\n"
                        f.write(line)
        
        print(f"✅ Tracking completato per {os.path.basename(output_path)}")

    def run_all(self):
        """Esegue il tracking su tutte le cartelle SNMOT-* trovate"""
        sequences = sorted(glob.glob(os.path.join(self.cfg.dataset_dir, 'SNMOT-*')))
        if not sequences:
            print(f"❌ Nessuna sequenza trovata in {self.cfg.dataset_dir}")
            return

        print(f"📋 Trovate {len(sequences)} sequenze.")
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            self.run_on_sequence(seq_name)
            # Reset modello per pulire la memoria dei track ID tra video
            self.model = YOLO(self.cfg.model_path)