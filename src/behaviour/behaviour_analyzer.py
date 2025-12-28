import os
import json
import numpy as np
from src.utils.bbox_operations import BBoxOperations


class BehaviorAnalyzer:
    def __init__(self, config):
        self.config = config
        self.roi_path = config['paths']['roi_config']

        # Risoluzione standard dataset SoccerNet/Challenge (come da slide)
        self.img_width = 1920
        self.img_height = 1080

        # Carica le ROI
        print(f"\nCaricamento configurazione ROI da: {self.roi_path}")
        with open(self.roi_path, 'r') as f:
            self.roi_data = json.load(f)

    def process_sequence(self, sequence_name):
        """
        Legge il file di tracking generato e calcola il behaviour.
        """
        # Recupera il percorso del file tracking generato dallo step precedente
        tracking_file = os.path.join(
            self.config['paths']['output_folder'],
            f"tracking_{sequence_name}_{self.config['settings']['team_id']}.txt"
        )

        output_file = os.path.join(
            self.config['paths']['output_folder'],
            f"behavior_{sequence_name}_{self.config['settings']['team_id']}.txt"
        )

        if not os.path.exists(tracking_file):
            print(f"Tracking file mancante per {sequence_name}. Salto behaviour.")
            return

        print(f"Analisi Behaviour su: {sequence_name}")

        # 1. Parsing delle ROI utilizzando BBoxOperations
        # Passiamo le dimensioni hardcoded (1920x1080) perché qui non carichiamo le immagini
        roi1_abs = BBoxOperations.get_absolute_roi(self.roi_data['roi1'], self.img_width, self.img_height)
        roi2_abs = BBoxOperations.get_absolute_roi(self.roi_data['roi2'], self.img_width, self.img_height)

        # 2. Caricamento dati tracking
        try:
            # Formato file tracking: frame, id, x, y, w, h
            data = np.loadtxt(tracking_file, delimiter=',')
        except Exception:
            data = np.empty((0, 6))

        if len(data) == 0:
            print(f"Nessun dato di tracking per {sequence_name}. Salto generazione behaviour.")
            open(output_file, 'w').close()
            return

        # Assicuriamoci che i dati siano float per i calcoli
        data = data.astype(float)

        # Ottieni i frame unici e ordinali
        unique_frames = np.unique(data[:, 0]).astype(int)

        with open(output_file, 'w') as f:
            for frame_idx in unique_frames:
                # Filtra le righe del frame corrente
                current_frame_dets = data[data[:, 0] == frame_idx]

                count_roi_1 = 0
                count_roi_2 = 0

                for det in current_frame_dets:
                    # Estrai coordinate bounding box: x, y, w, h
                    _, _, x, y, w, h = det

                    # LOGICA PAGINA 8:
                    # "A player is considered in a ROI if the center of the basis
                    # of the bounding box is inside the ROI"

                    # Usa la utility centrale per calcolare i piedi
                    feet_point = BBoxOperations.get_feet_point((x, y, w, h))

                    # Usa la utility centrale per verificare l'intersezione
                    if BBoxOperations.is_point_in_rect(roi1_abs, feet_point):
                        count_roi_1 += 1

                    if BBoxOperations.is_point_in_rect(roi2_abs, feet_point):
                        count_roi_2 += 1

                # Scrivi output come: frame_id, region_id, n_players
                f.write(f"{frame_idx},1,{count_roi_1}\n")
                f.write(f"{frame_idx},2,{count_roi_2}\n")

        print(f"Behaviour salvato: {output_file}")