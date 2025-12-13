import os
from ultralytics import YOLO

# --- CLASSE DI CONFIGURAZIONE ---
class TrackerConfig:
    def __init__(self, 
                 root_dir=None,
                 dataset_rel_path='tracking/train',
                 model_rel_path='models/player_detection_best.pt',
                 tracker_yaml_rel_path='src/tracker/botsort.yaml',
                 output_rel_path='output',
                 team_id="16",
                 img_size=1280,
                 conf_thresh=0.3,
                 classes_to_track=[1, 2, 3], # 0:ball, 1:gk, 2:player, 3:ref
                 limit_frames=None  # None per processare tutto
                ):
        
        # Imposta la root del progetto
        if root_dir is None:
            # Assume che lo script sia in una sottocartella, risale alla root
            self.root_dir = os.path.dirname(os.path.abspath(__file__))
            # Se lo script è nella root, usa quella, altrimenti aggiusta i ../ se necessario
            # Qui assumo che tu lanci lo script dalla root o da src
        else:
            self.root_dir = root_dir

        # Costruzione percorsi assoluti
        self.dataset_dir = os.path.join(self.root_dir, dataset_rel_path)
        self.model_path = os.path.join(self.root_dir, model_rel_path)
        self.tracker_config = os.path.join(self.root_dir, tracker_yaml_rel_path)
        self.output_dir = os.path.join(self.root_dir, output_rel_path)
        
        self.team_id = team_id
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.classes = classes_to_track
        self.limit_frames = limit_frames

        # Crea output dir se non esiste
        os.makedirs(self.output_dir, exist_ok=True)