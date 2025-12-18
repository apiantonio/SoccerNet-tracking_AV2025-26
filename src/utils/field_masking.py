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
    Genera maschera del campo con CLAHE e soglia adattiva.
    """
    # 1. Resize per velocità
    scale_factor = 1.0
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # 2. Converti in HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # CLAHE
    # Migliora il contrasto locale per gestire meglio ombre e nebbia.
    # Separiamo i canali
    h, s, v = cv2.split(hsv)
    
    # Creiamo l'oggetto CLAHE (Clip Limit evita di amplificare troppo il rumore)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Applichiamo SOLO al canale V (Luminosità) per evitare di alterare i colori
    v = clahe.apply(v)
    
    # Rimettiamo insieme l'immagine
    hsv = cv2.merge((h, s, v))
    
    # Logica dinamica per soglia verde
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    h_img, w_img = hsv.shape[:2]
    
    # Campiona ROI in basso al centro
    roi_y1 = int(h_img * 0.54)
    roi_y2 = int(h_img * 0.875) # 1/3 che parte da 1/8 dal basso
    roi_x1 = int(w_img * 0.17)
    roi_x2 = int(w_img * 0.83) 
    
    roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi.size > 0:
        median_hsv = np.median(roi, axis=(0, 1))
        
        # Sanity check per verificare che sia verde, se no mantieni soglia fissa
        if 30 < median_hsv[0] < 90:
            tol_h = 20
            tol_s = 60
            tol_v = 80
            
            lower_green = np.array([
                max(0, median_hsv[0] - tol_h),
                max(20, median_hsv[1] - tol_s),
                max(20, median_hsv[2] - tol_v)
            ])
            
            upper_green = np.array([
                min(180, median_hsv[0] + tol_h),
                min(255, median_hsv[1] + tol_s),
                min(255, median_hsv[2] + tol_v)
            ])

    # 3. Maschera
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 4. Pulizia
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close)
    
    # 5. Contorni e Hull
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    
    hull = (hull * (1 / scale_factor)).astype(np.int32)
    
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