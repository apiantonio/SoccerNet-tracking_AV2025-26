import os
import cv2
import json
import glob
from utils.bbox_operations import BBoxOperations
from utils.bbox_drawer import BBoxDrawer


class SoccerVisualizer:
    def __init__(self, config):
        """
        Inizializza il visualizzatore.
        config: dizionario caricato da yaml
        """
        self.config = config
        self.roi_path = config['paths']['roi_config']

        self.drawer = BBoxDrawer()

        # Carica configurazione ROI
        with open(self.roi_path, 'r') as f:
            self.roi_data = json.load(f)

    def _load_tracking_data(self, filepath):
        """Carica il file tracking_K_XX.txt"""
        data = {}
        if not os.path.exists(filepath):
            print(f"File tracking non trovato: {filepath}")
            return data

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                # Format: frame, id, x, y, w, h
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])

                if frame_idx not in data: data[frame_idx] = []
                data[frame_idx].append({'id': track_id, 'bbox': (int(x), int(y), int(w), int(h))})
        return data

    def _load_behavior_data(self, filepath):
        """Carica il file behavior_K_XX.txt"""
        # Format: frame, region_id, count
        data = {}
        if not os.path.exists(filepath):
            print(f"File behaviour non trovato: {filepath}")
            return data

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3: continue

                frame_idx = int(parts[0])
                region_id = int(parts[1])
                count = int(parts[2])

                if frame_idx not in data: data[frame_idx] = {}
                data[frame_idx][region_id] = count
        return data

    def generate_video(self, sequence_name):
        """Genera il video annotato per la sequenza data"""

        # Costruzione percorsi (LOGICA ORIGINALE)
        img_folder = os.path.join(self.config['paths']['input_folder'], sequence_name, "img1")
        output_dir = self.config['paths']['output_folder']
        team_id = self.config['settings']['team_id']

        track_file = os.path.join(output_dir, f"tracking_{sequence_name}_{team_id}.txt")
        behaviour_file = os.path.join(output_dir, f"behavior_{sequence_name}_{team_id}.txt")
        output_video_path = os.path.join(output_dir, f"video_{sequence_name}_{team_id}.mp4")

        # Caricamento Dati
        tracks = self._load_tracking_data(track_file)
        behaviors = self._load_behavior_data(behaviour_file)

        # Recupero Immagini
        img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
        if not img_files:
            print(f"Nessuna immagine trovata in {img_folder}")
            return

        # Setup Video Writer
        first_frame = cv2.imread(img_files[0])
        h_img, w_img, _ = first_frame.shape
        fps = 25
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_img, h_img))

        # Recupero ROI per questa sequenza utilizzando BBoxOperations
        roi1_rect = None
        if 'roi1' in self.roi_data:
            roi1_rect = BBoxOperations.get_absolute_roi(self.roi_data['roi1'], w_img, h_img)

        roi2_rect = None
        if 'roi2' in self.roi_data:
            roi2_rect = BBoxOperations.get_absolute_roi(self.roi_data['roi2'], w_img, h_img)

        print(f"Rendering Video: {sequence_name} -> {output_video_path}")

        for i, img_path in enumerate(img_files):
            frame = cv2.imread(img_path)
            frame_id = i + 1

            # --- VISUALIZZAZIONE ROI ---
            # Recupera conteggi per il frame corrente
            count1 = behaviors.get(frame_id, {}).get(1, 0)
            count2 = behaviors.get(frame_id, {}).get(2, 0)

            # Disegna ROI 1 (Rosso) tramite Drawer
            if roi1_rect:
                header_text = f"ROI 1 | Players: {count1}"
                self.drawer.draw_roi(frame, roi1_rect, header_text, 'roi1')

            # Disegna ROI 2 (Blu) tramite Drawer
            if roi2_rect:
                header_text = f"ROI 2 | Players: {count2}"
                self.drawer.draw_roi(frame, roi2_rect, header_text, 'roi2')

            # --- TRACKING E ID GIOCATORI ---
            if frame_id in tracks:
                for obj in tracks[frame_id]:
                    self.drawer.draw_player(frame, obj['bbox'], obj['id'])

            # --- INFO GENERALI ---
            info_text = f"Frame: {frame_id} | Total tracked: {len(tracks.get(frame_id, []))}"
            cv2.putText(frame, info_text, (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 1)

            out.write(frame)
            if i % 100 == 0:
                print(f"Processed frame {i}/{len(img_files)}")

        # --- CHIUSURA FUORI DAL CICLO ---
        out.release()
        print(f"Video completato: {output_video_path}")