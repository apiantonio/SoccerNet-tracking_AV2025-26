import os
import glob
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
# Percorso alla cartella che contiene tutte le sequenze (es. tracking/challenge o tracking/test)
DATASET_DIR = 'tracking/train' 

# Configurazione Tracker e Modello
TRACKER_CONFIG = 'src\\tracker\\botsort.yaml'
MODEL_PATH = 'models\\player_detection_best.pt'
OUTPUT_DIR = 'output'

# ID del tuo team
TEAM_ID = "16" 

def process_sequence(sequence_path, model, output_folder):
    """
    Processa una singola cartella SNMOT (es. SNMOT-061)
    """
    sequence_name = os.path.basename(sequence_path)
    
    # Costruisci il percorso alle immagini
    source_imgs = os.path.join(sequence_path, 'img1')
    
    # Controllo se esistono immagini
    if not os.path.exists(source_imgs):
        print(f"⚠️ Cartella immagini non trovata: {source_imgs}")
        return

    # Mappatura del nome file di output.
    # NOTA: Il PDF chiede tracking_K_XX.txt (K=VideoID, XX=TeamID).
    # Qui uso il nome della cartella originale (es. tracking_SNMOT-021_01.txt)
    # Dovrai rinominarli manualmente se il contest richiede ID numerici specifici (1, 2, 3...)
    output_filename = f"tracking_{sequence_name}_{TEAM_ID}.txt"
    output_path = os.path.join(output_folder, output_filename)

    print(f"🚀 Avvio Tracking su: {sequence_name}")
    
    # --- ESECUZIONE TRACKING SULL'INTERA CARTELLA ---
    # Passando il percorso della cartella, YOLO gestisce il loop interno
    # stream=True è importante per non saturare la RAM con lunghi video
    results = model.track(
        source=source_imgs, 
        persist=True, 
        tracker=TRACKER_CONFIG,
        conf=0.3,
        imgsz=1280, # O la dimensione che preferisci (1920 è nativa, ma più lenta)
        classes=[1, 2, 3], # Le classi sono: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        verbose=False,
        device=0 # Usa 'cpu' se non hai GPU
    )
    
    LIMIT_FRAMES = 750 #200 

    # Apriamo il file per salvare i risultati man mano che arrivano
    with open(output_path, 'w') as f:
        # Iteriamo sui risultati (YOLO qui agisce come un generatore frame per frame)
        for frame_idx, r in enumerate(results):
            
            # --- STOP HERE ---
            if frame_idx >= LIMIT_FRAMES:
                print(f"🛑 Test interrotto manualmente al frame {frame_idx}")
                break
            # -------------------
            
            # frame_id deve partire da 1 secondo le specifiche
            frame_id = frame_idx + 1 
            
            if r.boxes.id is not None:
                # Ottieni coordinate e ID
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes_xyxy, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # --- CONVERSIONE FORMATO secondo le specifiche ---
                    # Richiesto: frame_id, object_id, top_left_x, top_left_y, width, height
                    top_left_x = int(x1)
                    top_left_y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Scrittura riga
                    # Esempio PDF: 1,1,914,855,55,172
                    line = f"{frame_id},{track_id},{top_left_x},{top_left_y},{width},{height}\n"
                    f.write(line)
                    
    print(f"✅ Completato: {output_path}")

def main():
    # 1. Setup Modello
    model = YOLO(MODEL_PATH)
    
    # Crea cartella output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Trova tutte le cartelle SNMOT nella directory specificata
    # Cerca tutte le cartelle che iniziano con SNMOT
    sequences = sorted(glob.glob(os.path.join(DATASET_DIR, 'SNMOT-*')))
    
    if not sequences:
        print(f"❌ Nessuna sequenza trovata in {DATASET_DIR}")
        return

    print(f"Trovate {len(sequences)} sequenze da processare.")

    # 3. Ciclo su ogni video/sequenza
    for seq in sequences:
        process_sequence(seq, model, OUTPUT_DIR)
        
        # Reset del modello (opzionale ma consigliato tra video diversi per pulire la memoria dei track ID)
        model = YOLO(MODEL_PATH) 

if __name__ == "__main__":
    main()