import cv2
import numpy as np

SCALE_FACTOR = 0.5  # Fattore di riduzione per il calcolo della maschera
REL_X1 = 0.15  # Coordinate ROI per campionare il campo
REL_X2 = 0.85
REL_X3 = 0.40
REL_X4 = 0.60
REL_Y1 = 0.50
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
    
    # --- LOGICA ROI TRAPEZOIDALE ---
    # Creiamo una maschera binaria per definire DOVE campionare il colore
    h_img, w_img = hsv.shape[:2]
    roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    # Definiamo i 4 punti del trapezio (coordinate relative alla dimensione ridotta)
    # Lato corto in basso (per evitare pista/pubblicità negli angoli bassi)
    # Lato lungo in alto (per prendere più campo possibile)
    
    # Top-Left e Top-Right (più larghi, es. 15% - 85% width)
    tl = (int(w_img * REL_X1), int(h_img * REL_Y1)) 
    tr = (int(w_img * REL_X2), int(h_img * REL_Y1))
    
    # Bottom-Right e Bottom-Left (più stretti, es. 30% - 70% width)
    # Questo evita gli angoli in basso a sinistra/destra
    bl = (int(w_img * REL_X3), int(h_img * REL_Y2))
    br = (int(w_img * REL_X4), int(h_img * REL_Y2))
    
    pts = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.fillPoly(roi_mask, [pts], 255)
    
    # Estraiamo i pixel che cadono dentro il trapezio
    # hsv[roi_mask > 0] restituisce un array (N, 3) di pixel
    roi_pixels = hsv[roi_mask > 0]
    
    # Valori di default (fallback)
    lower_green = np.array([35, 45, 20])
    upper_green = np.array([85, 255, 255])
    
    if roi_pixels.size > 0:
        # Calcoliamo la mediana su tutti i pixel del trapezio
        median_hsv = np.median(roi_pixels, axis=0)
        
        # Sanity check: è verde?
        if 30 < median_hsv[0] < 90:
            tol_h = 18
            tol_s = 70
            
            # Logica "Shadow-Safe": 
            # Hue stretto, Saturation larga, Value COMPLETO (20-255) per accettare sole e ombra.
            lower_green = np.array([
                max(0, median_hsv[0] - tol_h),
                max(45, median_hsv[1] - tol_s),
                20  # V min fissa bassa (ombre)
            ])
            
            upper_green = np.array([
                min(180, median_hsv[0] + tol_h),
                min(255, median_hsv[1] + tol_s),
                255 # V max fissa alta (sole)
            ])

    # 3. Maschera
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Rimozione del bianco per i cartelloni e altre superfici riflettenti
    lower_white = np.array([0, 0, 180])   
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Sottraiamo i bianchi dalla maschera verde
    # (bitwise_not inverte white_mask, bitwise_and tiene solo ciò che è Verde E NON Bianco)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))
    
    # 4. Pulizia Morfologica (Dinamica in base allo SCALE_FACTOR)
    base_morph_size = 30 # Dimensione base
    k_size = max(3, int(base_morph_size * SCALE_FACTOR))
    
    # Erode piccolo per staccare elementi
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # Close grande per chiudere i giocatori e le linee bianche
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
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
        max_lift_y=40, 
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