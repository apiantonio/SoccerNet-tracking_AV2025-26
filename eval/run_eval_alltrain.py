import os
import yaml
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURAZIONE ---
# Se lo script è in eval/, ritorna alla cartella root
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)  # Cartella parent
os.chdir(root_dir)  # Cambia directory di lavoro

INPUT_ROOT = "tracking/train"    # Cartella dataset SoccerNet
MODEL_PATH = "models/player_detection_best.pt"  # Il tuo modello
TEMP_DATA_DIR = "temp_calibration_data"         # Cartella temporanea

# Mappa delle stringhe contenute nel gameinfo.ini alle classi YOLO
# 0: ball, 1: goalkeeper, 2: player, 3: referee
CLASS_MAPPING = {
    "ball": 0,
    "goalkeeper": 1,
    "player": 2,
    "referee": 3
}

def get_id_to_class_map(seq_path):
    """
    Legge gameinfo.ini e restituisce un dizionario {track_id: class_id}.
    Esempio: trackletID_14= referee;main  --> {14: 3}
    """
    ini_path = os.path.join(seq_path, "gameinfo.ini")
    id_map = {}
    
    if not os.path.exists(ini_path):
        print(f"⚠️ gameinfo.ini non trovato in {seq_path}")
        return id_map

    try:
        with open(ini_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Cerca righe che iniziano con trackletID_
                if line.startswith("trackletID_") and "=" in line:
                    key, val = line.split("=", 1)
                    val_lower = val.lower()
                    
                    # Estrai l'ID numerico (es. trackletID_14 -> 14)
                    try:
                        track_id = int(key.split("_")[1])
                    except (IndexError, ValueError):
                        continue

                    # Determina la classe in base al contenuto della stringa
                    detected_class = 2 # Default: player
                    for key_word, class_id in CLASS_MAPPING.items():
                        if key_word in val_lower:
                            detected_class = class_id
                            break
                    
                    id_map[track_id] = detected_class
    except Exception as e:
        print(f"⚠️ Errore parsing gameinfo.ini per {seq_path}: {e}")
        
    return id_map

def convert_mot_to_yolo(seq_name, input_root, output_root):
    """
    Converte gt.txt in etichette YOLO usando le classi corrette da gameinfo.ini.
    """
    seq_path = os.path.join(input_root, seq_name)
    img_dir = os.path.join(seq_path, "img1")
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    
    out_img_dir = os.path.join(output_root, "images", "val")
    out_lbl_dir = os.path.join(output_root, "labels", "val")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 1. Parsing delle classi per questa sequenza
    id_to_class = get_id_to_class_map(seq_path)

    # Leggi dimensioni immagine (dal primo frame)
    if not os.path.exists(img_dir): return 0
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    if not frames: return 0
    
    first_img_path = os.path.join(img_dir, frames[0])
    img = cv2.imread(first_img_path)
    if img is None: return 0
    img_h, img_w = img.shape[:2]

    # Leggi GT
    if not os.path.exists(gt_path): return 0
    try:
        gt_data = np.loadtxt(gt_path, delimiter=',') # [frame, id, x, y, w, h, conf, ...]
    except Exception: return 0
    if len(gt_data) == 0: return 0

    label_count = 0
    
    for filename in tqdm(frames, desc=f"Converting {seq_name}", leave=False):
        frame_idx = int(filename.split('.')[0])
        
        # Prefisso univoco per evitare sovrascritture tra sequenze
        new_filename = f"{seq_name}_{filename}"
        
        # Copia immagine
        src_img = os.path.join(img_dir, filename)
        dst_img = os.path.join(out_img_dir, new_filename)
        
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
            
        # Crea label file
        label_filename = new_filename.rsplit('.', 1)[0] + ".txt"
        label_file = os.path.join(out_lbl_dir, label_filename)
        
        # Filtra GT per il frame corrente
        frame_rows = gt_data[gt_data[:, 0] == frame_idx]
        
        with open(label_file, 'w') as f:
            for row in frame_rows:
                track_id = int(row[1])
                x, y, w, h = row[2], row[3], row[4], row[5]
                conf = row[6] # Solitamente 1 nel GT, ma controlliamo
                
                # Ignora annotazioni non sicure (opzionale, ma consigliato)
                # In SoccerNet GT a volte conf è 0 per oggetti occlusi/incerti? 
                # Solitamente nel GT ufficiale è 1.
                
                # Recupera la classe corretta dalla mappa. 
                # Se l'ID non è nel gameinfo (strano), fallback a 2 (player)
                cls_id = id_to_class.get(track_id, 2)
                
                # Converti in YOLO xywh normalizzato
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                
                # Clamp ai bordi
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0.001, min(1, nw))
                nh = max(0.001, min(1, nh))
                
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                label_count += 1
                
    return label_count

def create_dataset_yaml(data_path, yaml_path):
    """Crea il file dataset.yaml necessario a YOLO"""
    content = f"""
path: {os.path.abspath(data_path)} 
train: images/val  
val: images/val    

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
    # 0. Pulizia
    if os.path.exists(TEMP_DATA_DIR):
        print("🧹 Pulizia cartella temporanea...")
        shutil.rmtree(TEMP_DATA_DIR)

    # 1. Trova sequenze
    all_sequences = sorted([d for d in os.listdir(INPUT_ROOT) if d.startswith("SNMOT")])
    print(f"found {len(all_sequences)} sequences.")
    
    # 2. Conversione
    total_labels = 0
    print(f"\n🔄 Inizio conversione (Multi-classe da gameinfo.ini)...")
    
    for seq in all_sequences:
        count = convert_mot_to_yolo(seq, INPUT_ROOT, TEMP_DATA_DIR)
        total_labels += count
        
    print(f"\n✅ Conversione completata!")
    print(f"   Etichette generate: {total_labels}")

    # 3. YAML
    yaml_file = "calibration_dataset.yaml"
    create_dataset_yaml(TEMP_DATA_DIR, yaml_file)

    # 4. Validazione YOLO
    print("\n🚀 Avvio Analisi Curve Precision-Recall...")
    model = YOLO(MODEL_PATH)
    
    metrics = model.val(
        data=yaml_file,
        imgsz=1024,        
        conf=0.001,        # Confidenza minima per vedere tutta la curva
        iou=0.5,           # IOU match
        batch=16,
        plots=True,
        device=0,
        # Importante: ora possiamo validare su tutte le classi o solo quelle di interesse
        classes=[1, 2, 3]  # Escludiamo la palla (0) dai risultati se non ti interessa
    )

    print(f"\n📂 Grafici salvati in: {metrics.save_dir}")