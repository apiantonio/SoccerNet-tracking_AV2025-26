import cv2
import numpy as np

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
        # Usiamo una potenza cubica (3) invece di 1.5. 
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
    Genera maschera del campo.
    """
    # 1. OTTIMIZZAZIONE: Ridimensiona il frame per calcoli veloci
    # Lavorare su un'immagine più piccola velocizza inRange e findContours drasticamente
    scale_factor = 0.5 #1.0
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # 2. Converti in HSV (sull'immagine piccola)
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Range del verde
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 3. SEPARAZIONE TABELLONI (Erosione)
    # Prima di chiudere i buchi, erodiamo leggermente. Se il campo è attaccato 
    # a un tabellone pubblicitario da una linea sottile, questo la spezza.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    # 4. RIEMPIMENTO BUCHI (Closing)
    # Ora che i tabelloni sono staccati, chiudiamo i buchi dentro il campo (giocatori)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # Kernel proporzionato al resize
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close) # Rimuove rumore rimasto
    
    # 5. Trova contorni
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Ritorna maschera nera size originale se fallisce
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 6. Convex Hull
    hull = cv2.convexHull(largest_contour)
    
    # 7. RISCALA IL CONTORNO alle dimensioni originali
    hull = (hull * (1 / scale_factor)).astype(np.int32)
    
    # 8. RESTRINGIMENTO PROSPETTICO INTELLIGENTE
    # max_shrink_x: quanto stringere i lati (se non sono sul bordo video)
    # max_lift_y: quanto alzare il fondo (taglio netto pista/panchine)
    # border_thresh: tolleranza per considerare un punto "sul bordo video"
    hull = apply_perspective_shrink(
        hull, 
        frame.shape, 
        max_shrink_x=150, 
        max_lift_y=0, # Valore alto per tagliare bene il fondo sporco
        border_thresh=15
    )
    
    # Ricalcolo Hull finale per pulizia geometrica
    hull = cv2.convexHull(hull)
    
    # Disegna sulla maschera a grandezza originale
    clean_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.drawContours(clean_mask, [hull], -1, 255, thickness=cv2.FILLED)
    
    return clean_mask

def is_point_on_field(point, field_mask):
    """
    Controlla se un punto (x, y) è dentro la maschera del campo.
    """
    x, y = int(point[0]), int(point[1])
    h, w = field_mask.shape
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return field_mask[y, x] > 0