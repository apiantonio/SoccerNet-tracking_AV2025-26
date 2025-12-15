import os
import torch
import gc
import time 
import yaml
from ultralytics import YOLO

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

        self.device = config['tracker']['device']
        self.output_folder = config['paths']['output_folder']
        self.classes = config['tracker'].get('classes', [1,2,3])  # Default: traccia portieri(1) e giocatori(2) e arbitri(3)
        self.verbose = config['tracker'].get('verbose', False)
        
        # Parametri per ottimizzazione memoria
        self.imgsz = config['tracker'].get('imgsz', 960)
        self.half = config['tracker'].get('half', False)
    
        print(f"\n🔄 Caricamento Modello YOLO: {self.model_path}")
        self.model = YOLO(self.model_path)

    def track_sequence(self, sequence_name):
        """
        Esegue il tracking su una specifica sequenza (es. SNMOT-060)
        """
        # Costruisci i percorsi dinamicamente
        source_path = os.path.join(self.config['paths']['input_folder'], sequence_name, "img1")
        output_dir = self.config['paths']['output_folder']
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"tracking_{sequence_name}_{self.config['settings']['team_id']}.txt"
        output_path = os.path.join(output_dir, output_filename)

        print(f"🚀 Avvio Tracking: {sequence_name} | imgsz: {self.imgsz} | conf: {self.conf} | iou: {self.iou} | batch: {self.batch_size} | device: {self.device} | half-precision (FP16): {self.half} | verbose: {self.verbose}")

        # PULIZIA MEMORIA PRIMA DI INIZIARE
        gc.collect()
        torch.cuda.empty_cache()

        # Esecuzione Tracker
        current_time = time.time()
        results = self.model.track(
            source=source_path,
            persist=True,
            tracker=self.tracker_cfg_path,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            batch=self.batch_size,
            half=self.half,
            device=self.device,
            classes=self.classes, #  0:ball, 1:gk, 2:player, 3:ref
            verbose=self.verbose,
            stream=True,     # Stream è fondamentale per non saturare la RAM
        )
        elapsed_time = time.time() - current_time
        print(f"⏱️ Tracking completato in {elapsed_time:.2f} secondi.")
        
        # Scrittura Output
        current_time = time.time()
        with open(output_path, 'w') as f:
            for frame_idx, r in enumerate(results):
                if r.boxes is None or r.boxes.id is None:
                    continue
                
                boxes = r.boxes.xywh.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy()
                
                # Scriviamo nel formato richiesto: frame,id,x,y,w,h
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    # Converti da centro (xywh) a top-left (xywh)
                    x1 = x - (w / 2)
                    y1 = y - (h / 2)
                    
                    f.write(f"{frame_idx + 1},{int(track_id)},{int(x1)},{int(y1)},{int(w)},{int(h)}\n")
        
        print(f"💾 Salvato: {output_path}")
        elapsed_time = time.time() - current_time
        print(f"⏱️ Scrittura file completata in {elapsed_time:.2f} secondi.")
        
        self._copy_tracker_config()
        
        torch.cuda.empty_cache()
        return output_path
    
    def _copy_tracker_config(self):
        """
        Salva un file YAML nella cartella output che combina:
        1. La configurazione dell'algoritmo (es. botsort.yaml)
        2. I parametri di runtime sovrascritti da CLI (conf, iou, imgsz, ecc.)
        """
        if os.path.exists(self.tracker_cfg_path):
            try:
                # 1. Leggi la config base dell'algoritmo (es. BoT-SORT params)
                with open(self.tracker_cfg_path, 'r') as f:
                    final_tracker_cfg = yaml.safe_load(f) or {}

                # 2. Aggiungi/Sovrascrivi con i parametri di runtime effettivi (da self.config)
                # Questi sono i valori REALI usati nello script, inclusi gli override CLI
                runtime_overrides = {
                    'inference_settings': {
                        'conf': self.conf,
                        'iou': self.iou,
                        'imgsz': self.imgsz,
                        'device': self.device,
                        'batch_size': self.batch_size,
                        'half': self.half,
                        'classes': self.classes
                    }
                }
                
                # Uniamo tutto
                final_tracker_cfg.update(runtime_overrides)

                # 3. Scrivi il nuovo file YAML "freezato"
                cfg_filename = os.path.basename(self.tracker_cfg_path)
                self.dst_cfg_path = os.path.join(self.output_folder, cfg_filename)

                with open(self.dst_cfg_path, 'w') as f:
                    yaml.dump(final_tracker_cfg, f, sort_keys=False)
                
                print(f"📋 Configurazione Tracker (Runtime + Algo) salvata in: {self.dst_cfg_path}")
                
            except Exception as e:
                print(f"⚠️ Errore salvataggio config tracker: {e}")
        else:
            print(f"⚠️ File config tracker originale non trovato: {self.tracker_cfg_path}")