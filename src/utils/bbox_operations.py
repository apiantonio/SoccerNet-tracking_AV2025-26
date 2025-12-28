class BBoxOperations:
    """
    Classe di utilità per operazioni geometriche su Bounding Box.
    Gestisce conversioni di formato e calcoli spaziali.
    """

    @staticmethod
    def get_absolute_roi(rel_rect, img_w, img_h):
        """
        Converte un rettangolo con coordinate relative (0-1) in pixel assoluti.

        Args:
            rel_rect (dict): Dizionario con chiavi 'x', 'y', 'width', 'height' (range di valori 0.0-1.0).
            img_w (int): Larghezza immagine.
            img_h (int): Altezza immagine.

        Returns:
            tuple: (x, y, w, h) in pixel interi.
        """
        x = int(rel_rect['x'] * img_w)
        y = int(rel_rect['y'] * img_h)
        w = int(rel_rect['width'] * img_w)
        h = int(rel_rect['height'] * img_h)
        return x, y, w, h

    @staticmethod
    def is_point_in_rect(rect, point):
        """
        Verifica se un punto è contenuto all'interno di un rettangolo.

        Args:
            rect (tuple): (x, y, w, h) coordinate Top-Left.
            point (tuple): (px, py) coordinate del punto.

        Returns:
            bool: True se il punto è dentro, False altrimenti.
        """
        rx, ry, rw, rh = rect
        px, py = point
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    @staticmethod
    def get_feet_point(bbox_xywh):
        """
        Calcola il punto centrale della base (piedi) di un bounding box.

        Args:
            bbox_xywh (tuple): (x, y, w, h) coordinate Top-Left.

        Returns:
            tuple: (x, y) coordinate dei piedi.
        """
        x, y, w, h = bbox_xywh
        return int(x + w / 2), int(y + h)

    @staticmethod
    def center_to_top_left(xywh):
        """
        Converte un bounding box dal formato YOLO (Centro) al formato standard (Top-Left).

        Args:
            xywh (tuple): (center_x, center_y, width, height).

        Returns:
            tuple: (x, y, w, h) coordinate Top-Left.
        """
        cx, cy, w, h = xywh
        return int(cx - w / 2), int(cy - h / 2), int(w), int(h)