import cv2
import numpy as np


class BBoxDrawer:
    """
    Gestisce il disegno di Bounding Box, ROI.
    """

    PALETTE = {
        'roi1': (0, 0, 255),  # Rosso
        'roi2': (255, 0, 0),  # Blu
        'bg_tag': (40, 40, 40),  # Grigio scuro (sfondo label)
        'text': (255, 255, 255)  # Bianco
    }

    def __init__(self):
        self._color_cache = {}

    def get_id_color(self, entity_id):
        """Restituisce un colore consistente per un dato ID."""
        if entity_id not in self._color_cache:
            np.random.seed(int(entity_id))
            self._color_cache[entity_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self._color_cache[entity_id]

    def draw_roi(self, img, rect, label_text, color_key='roi1'):
        """Disegna una ROI semitrasparente con etichetta."""
        x, y, w, h = rect
        color = self.PALETTE.get(color_key, (0, 255, 0))

        # Overlay semitrasparente
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Bordo
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Label
        self.draw_tag(img, label_text, (x + w // 2, y - 15), bg_color=color)

    def draw_player(self, img, bbox, id):
        """Disegna box giocatore, piedi e ID."""
        x, y, w, h = map(int, bbox)
        color = self.get_id_color(id)

        # Box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Piedi
        feet_x, feet_y = int(x + w // 2), int(y + h)
        cv2.circle(img, (feet_x, feet_y), 4, color, -1)
        cv2.circle(img, (feet_x, feet_y), 5, (0, 0, 0), 1)  # Bordo nero

        # Label
        self.draw_tag(img, f"ID {id}", (feet_x, feet_y + 12))

    def draw_tag(self, img, text, anchor_center, bg_color=None):
        """Disegna etichetta testo con sfondo."""
        if bg_color is None:
            bg_color = self.PALETTE['bg_tag']

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.5
        thick = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thick)

        cx, cy = anchor_center
        pad = 6
        x1, y1 = cx - text_w // 2 - pad, cy - text_h // 2 - pad
        x2, y2 = cx + text_w // 2 + pad, cy + text_h // 2 + pad

        # Verifica bordi immagine
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Sfondo label
        sub = img[y1:y2, x1:x2]
        if sub.size > 0:
            rect = np.full_like(sub, bg_color)
            cv2.addWeighted(sub, 0.4, rect, 0.6, 0, sub)
            img[y1:y2, x1:x2] = sub

        # Testo
        cv2.putText(img, text, (x1 + pad, y2 - pad), font, scale, self.PALETTE['text'], thick, cv2.LINE_AA)