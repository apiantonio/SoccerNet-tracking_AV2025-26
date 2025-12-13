import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
SOURCE_VIDEO = 'test_video_input.mp4'
OUTPUT_VIDEO = 'output_tracking_test_stride5.mp4'
TRACKER_CONFIG = 'src\\tracker\\botsort.yaml' # Usa il tuo file config

# Carica modello
model = YOLO('models\player_detection_best.pt')
print(f"🖥️ Il modello sta girando su: {model.device}")

# Inizializza Video Capture
cap = cv2.VideoCapture(SOURCE_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Inizializza Video Writer
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"🚀 Avvio elaborazione video: {SOURCE_VIDEO} con BoT-SORT")

# Definiamo colori fissi per le classi (BGR)
COLORS = {
    'Player': (0, 0, 255),  # Rosso
    'Ref': (0, 255, 255),   # Giallo
    'GK': (0, 255, 0),      # Verde
    'Ball': (255, 0, 0)     # Blu
}

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    if frame_count % 50 == 0: print(f"Processing frame {frame_count}...")

    # 1. TRACKING con BoT-SORT
    # Passiamo il path del tuo file yaml custom al parametro 'tracker'
    results = model.track(
        frame, 
        device='cuda',    # Usa 'cpu' o 'cuda' se hai una GPU compatibile
        persist=True, 
        tracker=TRACKER_CONFIG,
        conf=0.3,         # Accetta anche giocatori meno visibili
        imgsz=1280,       # Usa alta risoluzione per vedere meglio la palla/giocatori lontani
        classes=[1, 2, 3], # Le classi sono: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        vid_stride=5,
        #verbose=True
    )
    
    # Se non ci sono ID tracciati (nessun oggetto o tracking fallito), salva frame pulito
    if results[0].boxes.id is None:
        out.write(frame)
        continue

    # Estrai dati dal risultato (spostati su CPU e convertiti in numpy)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.int().cpu().numpy()
    clss = results[0].boxes.cls.int().cpu().numpy()

    # Iteriamo su tutti gli oggetti rilevati e tracciati
    for box, track_id, cls in zip(boxes, ids, clss):
        x1, y1, x2, y2 = map(int, box)
        
        if cls == 2: # Player
            # Disegna Player (Verde)
            color = COLORS['Player']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label semplificata: Solo ID
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        elif cls == 3: # Referee
            # Disegna Arbitro (Giallo) - Utile per verificare che vengano distinti
            color = COLORS['Ref']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "REF", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        elif cls == 0: # Ball
            # Disegna Palla (Bianco)
            color = COLORS['Ball']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Scrivi il frame annotato
    out.write(frame)

cap.release()
out.release()
print(f"✅ Elaborazione completata! Video salvato in: {OUTPUT_VIDEO}")