import os
import glob
from tracker_config import TrackerConfig
from tracker.soccer_tracker import SoccerTracker
from visualizer.tracking_visualizer import TrackingVisualizer
import torch
import gc

def main():
    # 1. SETUP CONFIGURAZIONE
    # Puoi modificare questi parametri o passarli via argparse se vuoi espandere
    current_dir = os.getcwd() # O imposta manuale: "C:/Progetti/SoccerNet"
    
    config = TrackerConfig(
        root_dir=current_dir,
        dataset_rel_path='tracking/train',
        output_rel_path='output/buffer60_match70_high60_reid',
        model_rel_path='models/player_detection_best.pt',
        tracker_yaml_rel_path='src/tracker/botsort.yaml',
        team_id="16",
        limit_frames=None, # Metti un numero (es. 200) per test veloci, None per tutto
        classes_to_track=[1, 2, 3], # 0:ball, 1:gk, 2:player, 3:ref
        img_size=1280,     # Usa 1280 o 1088 per bilanciare qualità/velocità
        conf_thresh=0.3,
    )

    # Pulisci memoria GPU prima di iniziare
    gc.collect()
    torch.cuda.empty_cache()

    # 2. ISTANZIA LE CLASSI
    tracker = SoccerTracker(config)
    visualizer = TrackingVisualizer(config)

    # 3. SCELTA OPERAZIONE
    # Esempio A: Esegui su una singola sequenza specifica e poi visualizza
    target_seq = "SNMOT-060"
    
    # Step 3.1: Tracking
    txt_output = tracker.run_on_sequence(target_seq)
    
    # Step 3.2: Visualizzazione (solo se il tracking è andato a buon fine)
    if txt_output:
        visualizer.create_video(target_seq, tracking_file=txt_output)

    # Esempio B: Esegui su TUTTE le sequenze (commenta la parte sopra e usa questa)
    # tracker.run_all()
    
    # Se vuoi visualizzare un tracking già fatto senza rifarlo:
    # visualizer.create_video("SNMOT-060", tracking_file="path/to/existing.txt")

if __name__ == "__main__":
    main()