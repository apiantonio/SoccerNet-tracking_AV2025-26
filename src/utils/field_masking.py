import cv2
import numpy as np

SCALE_FACTOR = 1.0  # Fattore di riduzione per il calcolo della maschera
REL_X1 = 0.15  # Coordinate ROI per campionare il campo
REL_X2 = 0.85
REL_Y1 = 0.55
REL_Y2 = 0.90

def apply_perspective_shrink(contour, image_shape, max_shrink_x=120, max_lift_y=80, border_thresh=10):
    """
    Restringe il contorno dinamicamente.
    - NON tocca i punti che sono sui bordi laterali del video (es. inquadratura parziale).
    - Taglia aggressivamente il fondo (pista/panchine).
    """
    h_img, w_img = image_shape[:2]
    
    # 1. Trova il centroide del poligono
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
    else:
        cX = w_img // 2 
    
    new_contour_points = []
    
    for point in contour[:, 0, :]:
        x, y = point
        
        # --- LOGICA 1: Calcolo Fattore Y (Aggressività sul fondo) ---
        # Usiamo una curva quadratica per aumentare l'effetto verso il fondo.
        # Questo significa che l'effetto è quasi nullo a metà campo e fortissimo solo alla fine.
        y_rel = y / h_img
        y_factor = y_rel ** 2 
        
        # --- LOGICA 2: Protezione Bordi Video ---
        # Se il punto è molto vicino al bordo sinistro (0) o destro (w_img), 
        # assumiamo che il campo continui fuori inquadratura, quindi NON stringiamo X.
        is_on_edge = (x < border_thresh) or (x > w_img - border_thresh)
        
        # Calcolo spostamenti
        if is_on_edge:
            shift_x = 0  # Non spostare orizzontalmente se è sul bordo
        else:
            shift_x = int(max_shrink_x * y_factor)
            
        shift_y = int(max_lift_y * y_factor)
        
        # Applica spostamento X verso il centro
        new_x = x
        if not is_on_edge:
            if x < cX: 
                new_x = min(x + shift_x, cX)
            else:
                new_x = max(x - shift_x, cX)
            
        # Applica spostamento Y (solo verso l'alto per tagliare il fondo)
        new_y = max(0, y - shift_y)
        
        new_contour_points.append([new_x, new_y])
    
    return np.array(new_contour_points, dtype=np.int32).reshape((-1, 1, 2))

def get_field_mask(frame):
    """
    Genera maschera del campo robusta a ombre e luci.
    """
    # 1. SCALE FACTOR: Mantenerlo a 0.5 è cruciale per le performance morfologiche.
    # Se lo metti a 1.0, dovresti raddoppiare le dimensioni dei kernel (es. (30,30))
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    
    # 2. Converti in HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # CLAHE sul canale V (Luminosità)
    # Aiuta a "schiarire" le ombre profonde prima della sogliatura
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # ClipLimit alzato leggermente
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))
    
    # Calcola intervallo verde dinamico
    lower_green = np.array([35, 40, 20]) # V basso di default
    upper_green = np.array([85, 255, 255])
    
    h_img, w_img = hsv.shape[:2]
    
    # ROI: Campiona il campo
    roi_y1 = int(h_img * REL_Y1)
    roi_y2 = int(h_img * REL_Y2)
    roi_x1 = int(w_img * REL_X1)
    roi_x2 = int(w_img * REL_X2)
    
    roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi.size > 0:
        median_hsv = np.median(roi, axis=(0, 1))
        
        # Sanity check per controllare se il campo è verde
        if 30 < median_hsv[0] < 90:
            # Tonalità (H): Stretta attorno alla mediana per non prendere altri colori
            tol_h = 20
            
            # Saturazione (S): Abbastanza larga per gestire zone sbiadite
            tol_s = 70
            
            # Luminosità (V): QUI STA IL TRUCCO PER LE OMBRE.
            # Non usiamo la tolleranza sulla mediana per V.
            # Impostiamo un range fisso molto ampio.
            # Min: 20 (quasi nero, per prendere le ombre scure)
            # Max: 255 (bianco, per prendere le zone al sole pieno)
            # L'idea è: "Se la Tinta è verde, non mi importa se è scuro o chiaro".
            
            lower_green = np.array([
                max(0, median_hsv[0] - tol_h),
                max(20, median_hsv[1] - tol_s), # S minima 20 per evitare grigi
                20  # V minima fissa e bassa (accetta ombre)
            ])
            
            upper_green = np.array([
                min(180, median_hsv[0] + tol_h),
                min(255, median_hsv[1] + tol_s),
                255 # V massima fissa (accetta sole)
            ])

    # 3. Maschera
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 4. Pulizia Morfologica
    kernel_scale = int(SCALE_FACTOR*2) # Adatta i kernel alla scala ridotta (es. 0.5 -> kernel normale, 1.0 -> kernel*2)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_scale*3, kernel_scale*3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    # Closing per chiudere i giocatori (buchi)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_scale*15, kernel_scale*15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # Open per rimuovere rumore esterno (spalti)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close)
    
    # 5. Contorni e Hull
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    # Riportiamo alla scala originale
    hull = (hull * (1 / SCALE_FACTOR)).astype(np.int32)
    
    # Shrink prospettico
    hull = apply_perspective_shrink(
        hull, 
        frame.shape, 
        max_shrink_x=130, 
        max_lift_y=0, 
        border_thresh=15
    )
    
    hull = cv2.convexHull(hull)
    
    clean_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.drawContours(clean_mask, [hull], -1, 255, thickness=cv2.FILLED)
    
    return clean_mask

def is_point_on_field(point, field_mask, bottom_tolerance=40):
    """
    Controlla se un punto è nel campo.
    Include una logica di sicurezza per il bordo inferiore.
    
    :param bottom_tolerance: Se il punto è negli ultimi N pixel in basso, 
                             controlla la maschera un po' più in su.
    """
    x, y = int(point[0]), int(point[1])
    h, w = field_mask.shape
    
    # Clamp x per sicurezza
    x = max(0, min(x, w - 1))
    
    # Se il giocatore tocca il fondo spostiamo il punto di controllo verso l'alto per evitare falsi negativi
    if y >= h - bottom_tolerance:
        y_check = h - bottom_tolerance - 1
        # Assicuriamoci di non andare sotto zero
        y_check = max(0, y_check)
        return field_mask[y_check, x] > 0

    # Controllo standard per il resto dell'immagine
    y = max(0, min(y, h - 1))
    return field_mask[y, x] > 0