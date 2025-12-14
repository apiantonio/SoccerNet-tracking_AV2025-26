import os
import json
import numpy as np

class BehaviorAnalyzer:
    def __init__(self, config):
        self.config = config
        self.roi_path = config['paths']['roi_config']
        
        # Risoluzione standard dataset SoccerNet/Challenge (come da slide)
        self.img_width = 1920
        self.img_height = 1080
        
        # Carica le ROI
        print(f"\n📖 Caricamento configurazione ROI da: {self.roi_path}")
        with open(self.roi_path, 'r') as f:
            self.roi_data = json.load(f)
        
    def _get_absolute_roi(self, roi_relative):
        """
        Converte le coordinate ROI relative (0-1) in assolute (pixel).
        Formato input: {'x': 0.1, 'y': 0.2, 'width': 0.4, 'height': 0.4}
        Formato output: (x_min, y_min, x_max, y_max)
        """
        x = roi_relative['x'] * self.img_width
        y = roi_relative['y'] * self.img_height
        w = roi_relative['width'] * self.img_width
        h = roi_relative['height'] * self.img_height
        
        return x, y, x + w, y + h

    def _is_point_in_rect(self, point, rect):
        """
        Verifica se un punto (px, py) è dentro un rettangolo (x1, y1, x2, y2).
        """
        px, py = point
        rx1, ry1, rx2, ry2 = rect
        return rx1 <= px <= rx2 and ry1 <= py <= ry2

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
            print(f"⚠️ Tracking file mancante per {sequence_name}. Salto behaviour.")
            return

        print(f"🧠 Analisi Behaviour su: {sequence_name}")

        # 1. Parsing delle ROI
        roi1_abs = self._get_absolute_roi(self.roi_data['roi1'])
        roi2_abs = self._get_absolute_roi(self.roi_data['roi2'])

        # 2. Caricamento dati tracking
        try:
            # Formato file tracking: frame, id, x, y, w, h
            data = np.loadtxt(tracking_file, delimiter=',')
        except Exception:
            data = np.empty((0, 6))

        if len(data) == 0:
            print(f"⚠️ Nessun dato di tracking nel file. Genero file behaviour vuoto.")
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
                    
                    center_basis_x = x + (w / 2.0)
                    center_basis_y = y + h
                    
                    point = (center_basis_x, center_basis_y)
                    
                    if self._is_point_in_rect(point, roi1_abs):
                        count_roi_1 += 1
                    
                    if self._is_point_in_rect(point, roi2_abs):
                        count_roi_2 += 1
                
                # Scrivi output come da Pagina 9: frame_id, region_id, n_players
                f.write(f"{frame_idx},1,{count_roi_1}\n")
                f.write(f"{frame_idx},2,{count_roi_2}\n")

        print(f"✅ Behaviour salvato: {output_file}")