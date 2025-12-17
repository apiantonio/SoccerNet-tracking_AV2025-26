import os
import yaml
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from IPython.display import Image, display

# --- CONFIGURAZIONE ---
# Se lo script è in eval/, ritorna alla cartella root
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)  # Cartella parent
os.chdir(root_dir)  # Cambia directory di lavoro

SEQ_NAME = "SNMOT-060"           # Sequenza da usare per la calibrazione
INPUT_ROOT = "tracking/train"    # Cartella dove hai scaricato il dataset
MODEL_PATH = "models/player_detection_best.pt"  # Il tuo modello
TEMP_DATA_DIR = "temp_calibration_data"         # Cartella temporanea per i dati convertiti

# Mappa classi (dal tuo main_config.yaml: 0=ball, 1=gk, 2=player, 3=ref)
# NOTA: Assicurati che il tuo modello usi questi stessi indici!
CLASS_MAP = {1: 1, 2: 2, 3: 3} 

def convert_mot_to_yolo(seq_name, input_root, output_root):
    """Converte gt.txt in etichette YOLO per la validazione"""
    
    # Percorsi
    img_dir = os.path.join(input_root, seq_name, "img1")
    gt_path = os.path.join(input_root, seq_name, "gt", "gt.txt")
    
    out_img_dir = os.path.join(output_root, "images", "val")
    out_lbl_dir = os.path.join(output_root, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"🔄 Conversione GT per {seq_name}...")
    
    # Leggi dimensioni immagine (dal primo frame)
    first_img = os.listdir(img_dir)[0]
    img = cv2.imread(os.path.join(img_dir, first_img))
    img_h, img_w = img.shape[:2]

    # Leggi GT
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT non trovato: {gt_path}")
        
    gt_data = np.loadtxt(gt_path, delimiter=',')
    
    # Copia immagini e crea label
    frames = sorted(os.listdir(img_dir))
    # Limitiamo a un sottoinsieme se vuoi fare veloce (es. primi 200 frame), altrimenti tutto
    # frames = frames[:300] 
    
    label_count = 0
    for filename in tqdm(frames):
        if not filename.endswith(('.jpg', '.png')): continue
        
        frame_idx = int(filename.split('.')[0])
        
        # Copia immagine
        src_img = os.path.join(img_dir, filename)
        dst_img = os.path.join(out_img_dir, filename)
        if not os.path.exists(dst_img):
            shutil.copy2(os.path.abspath(src_img), dst_img)
            
        # Crea label file
        label_file = os.path.join(out_lbl_dir, filename.replace(filename.split('.')[-1], 'txt'))
        
        # Filtra righe GT per questo frame
        frame_rows = gt_data[gt_data[:, 0] == frame_idx]
        
        with open(label_file, 'w') as f:
            for row in frame_rows:
                # MOT format SoccerNet: frame, id, x, y, w, h, conf, -1, -1, -1
                # Le classi NON sono nel gt.txt - assegniamo tutto come "player" (cls 2)
                # Potrai affinare dopo con annotazioni separate se necessario
                cls_id = 2  # player
                
                x, y, w, h = row[2], row[3], row[4], row[5]
                conf = row[6]
                
                # Filtra le detection con conf bassa
                if conf < 0.5:
                    continue
                
                # Converti in YOLO xywh normalizzato
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # Clamp ai bordi immagine
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0.001, min(1, nw))
                nh = max(0.001, min(1, nh))
                
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                label_count += 1
    
    print(f"✅ Copiate {len(frames)} immagini, create {label_count} etichette")

    return out_img_dir

def create_dataset_yaml(data_path, yaml_path):
    """Crea il file dataset.yaml necessario a YOLO"""
    content = f"""
path: {os.path.abspath(data_path)} # dataset root dir
train: images/val  # usiamo val anche per train per finta
val: images/val    # immagini di validazione
test:              # test images (optional)

# Classes
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"✅ Creato file config: {yaml_path}")

# --- MAIN ---
if __name__ == '__main__':
    # 1. Prepara i dati
    if os.path.exists(TEMP_DATA_DIR): shutil.rmtree(TEMP_DATA_DIR)
    convert_mot_to_yolo(SEQ_NAME, INPUT_ROOT, TEMP_DATA_DIR)

    # 2. Crea yaml
    yaml_file = "calibration_dataset.yaml"
    create_dataset_yaml(TEMP_DATA_DIR, yaml_file)

    # 3. Esegui model.val()
    print("\n🚀 Avvio Analisi Curve Precision-Recall...")
    model = YOLO(MODEL_PATH)
    metrics = model.val(
        data=yaml_file,
        imgsz=1024,        # Risoluzione input (come da tuo config)
        conf=0.1,        # Confidenza minima per vedere tutta la curva
        iou=0.3,           # NMS IoU (0.7 consigliato per calcio)
        batch=16,
        plots=True,
        device=0,
        classes=[1,2,3]
    )

    # 4. Mostra i risultati
    save_dir = metrics.save_dir
    print(f"\n📂 Grafici salvati in: {save_dir}")