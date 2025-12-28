import cv2
import numpy as np

# Fattore di scala per velocizzare l'elaborazione (0.5 = metà risoluzione)
SCALE_FACTOR = 0.5


def get_field_mask(frame):
    """
    Genera una maschera del campo robusta utilizzando:
    1. Analisi istogramma adattiva (invece di ROI fissa)
    2. Combinazione spazi colore HSV + LAB (canale 'a' per il verde)
    3. FloodFill e ConvexHull
    """
    h_orig, w_orig = frame.shape[:2]

    # 1. Resize per performance
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    h_img, w_img = small_frame.shape[:2]

    # Pre-processing: Blur leggero per ridurre rumore dell'erba
    blurred = cv2.GaussianBlur(small_frame, (5, 5), 0)

    # ---------------------------------------------------------
    # STEP 1: Rilevamento Colore Adattivo (Istogramma)
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Analizziamo solo la metà inferiore dell'immagine per trovare il colore del campo
    # (evitiamo di analizzare il cielo o gli spalti alti)
    roi_hist = hsv[int(h_img * 0.4):, :, 0]  # Canale Hue, dal 40% in giù

    # Calcolo istogramma della Hue (0-180)
    hist = cv2.calcHist([roi_hist], [0], None, [180], [0, 180])

    # Cerchiamo il picco nell'intervallo del verde (circa 30-90 in OpenCV Hue)
    # Azzeriamo tutto ciò che non è verde verosimile per trovare il picco giusto
    hist[:30] = 0
    hist[95:] = 0

    peak_hue = np.argmax(hist)

    # Se non troviamo un picco valido (es. inquadratura solo pubblico), fallback a default
    if peak_hue == 0:
        peak_hue = 60  # Verde standard

    # Definiamo tolleranze dinamiche
    hue_tol = 18  # Tolleranza colore
    sat_min = 35  # Saturazione minima (evita grigi/bianchi)
    val_min = 30  # Luminosità minima (evita neri assoluti)

    lower_green_hsv = np.array([max(0, peak_hue - hue_tol), sat_min, val_min])
    upper_green_hsv = np.array([min(180, peak_hue + hue_tol), 255, 255])

    mask_hsv = cv2.inRange(hsv, lower_green_hsv, upper_green_hsv)

    # ---------------------------------------------------------
    # STEP 2: Refinement con Spazio LAB (Canale 'a')
    # ---------------------------------------------------------
    # Il canale 'a' in LAB va da Verde (valori bassi) a Rosso (valori alti).
    # In OpenCV uint8, 128 è neutro. Valori < 120 sono fortemente verdi.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Soglia empirica robusta per l'erba: 'a' deve essere basso (verde)
    # L'erba solitamente sta sotto 115-120 su scala 0-255
    mask_lab = cv2.inRange(lab, np.array([0, 0, 0]), np.array([255, 118, 255]))

    # Intersezione: deve essere verde SIA in HSV SIA in LAB
    combined_mask = cv2.bitwise_and(mask_hsv, mask_lab)

    # ---------------------------------------------------------
    # STEP 3: Pulizia Morfologica
    # ---------------------------------------------------------
    # Kernel ellittici
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    # Rimuovi rumore (spettatori con maglie verdi, coriandoli)
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k_open, iterations=2)

    # Chiudi i buchi (giocatori, linee bianche interne al campo)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, k_close, iterations=2)

    # ---------------------------------------------------------
    # STEP 4: Selezione dell'Area Maggiore (FloodFill / Contorni)
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(mask_clean)

    if contours:
        # Trova il contorno più grande (sicuramente il campo)
        largest_contour = max(contours, key=cv2.contourArea)

        # Se il contorno è troppo piccolo (es. inquadratura zoomata sul pubblico), ritorna vuoto
        if cv2.contourArea(largest_contour) < (h_img * w_img * 0.05):
            return cv2.resize(final_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Calcola Convex Hull per regolarizzare la forma (il campo è convesso)
        hull = cv2.convexHull(largest_contour)

        # --- Ottimizzazione Bordi ---
        # Spesso la hull include angoli di spalti in alto.
        # Possiamo migliorare approssimando a un poligono con meno vertici
        epsilon = 0.005 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)

    # ---------------------------------------------------------
    # STEP 5: Post-Processing e Resize
    # ---------------------------------------------------------
    # Riporta alla dimensione originale
    final_mask_full = cv2.resize(final_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    return final_mask_full


def is_point_on_field(point, field_mask, bottom_tolerance=40):
    """
    Controlla se un punto è nel campo.
    """
    x, y = int(point[0]), int(point[1])
    h, w = field_mask.shape

    # Clamp coordinates
    x = max(0, min(x, w - 1))

    # Tolleranza per i piedi che toccano quasi il bordo inferiore dell'immagine
    # Se siamo molto in basso, assumiamo sia campo (la camera non inquadra sotto terra)
    if y >= h - bottom_tolerance:
        return True

    y = max(0, min(y, h - 1))

    # Verifica il valore della maschera (255 = campo)
    return field_mask[y, x] > 127