import json
import os
import cv2
import numpy as np
import glob

class BehaviorAnalyzer:
    def __init__(self, config_json_path, tracking_file_path, output_folder, team_id="16"):
        """
        Args:
            config_json_path: Path al file JSON con le ROI (roi1, roi2).
            tracking_file_path: Path al file .txt del tracking generato in precedenza.
            output_folder: Dove salvare i risultati e il video.
            team_id: ID del team per il nome del file (default "16").
        """
        self.json_path = config_json_path
        self.tracking_path = tracking_file_path
        self.output_folder = output_folder
        self.team_id = team_id
        
        # Dati interni
        self.rois = {}     # { 'roi1': [x, y, w, h], 'roi2': ... } (in pixel)
        self.tracks = {}   # { frame_id: [ {id, box, foot_point}, ... ] }
        self.counts = {}   # { frame_id: { 'roi1': count, 'roi2': count } }
        
        # Carica subito i dati
        self._load_tracking_data()

    def _load_config(self, img_w, img_h):
        """Carica il JSON e converte coordinate relative (0-1) in assolute (pixel)"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Config JSON non trovato: {self.json_path}")
            
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        # Conversione
        for roi_key in ['roi1', 'roi2']:
            if roi_key in data:
                r = data[roi_key]
                # PDF: valori relativi * dimensione immagine
                abs_x = int(r['x'] * img_w)
                abs_y = int(r['y'] * img_h)
                abs_w = int(r['width'] * img_w)
                abs_h = int(r['height'] * img_h)
                self.rois[roi_key] = (abs_x, abs_y, abs_w, abs_h)
        
        print(f"📐 ROI caricate e scalate su {img_w}x{img_h}: {self.rois}")

    def _load_tracking_data(self):
        """Carica il file tracking_K_XX.txt"""
        if not os.path.exists(self.tracking_path):
            raise FileNotFoundError(f"Tracking file mancante: {self.tracking_path}")

        print(f"📂 Caricamento tracking: {self.tracking_path}")
        with open(self.tracking_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # Formato atteso: frame, id, x, y, w, h
                if len(parts) < 6: continue
                
                fid = int(parts[0])
                tid = int(parts[1])
                x, y = int(float(parts[2])), int(float(parts[3]))
                w, h = int(float(parts[4])), int(float(parts[5]))
                
                # Calcolo "Center of Basis" (Piede) come da PDF
                # Center x = x + w/2
                # Bottom y = y + h
                foot_x = int(x + w / 2)
                foot_y = int(y + h)
                
                if fid not in self.tracks: self.tracks[fid] = []
                self.tracks[fid].append({
                    'id': tid,
                    'bbox': (x, y, w, h),
                    'foot': (foot_x, foot_y)
                })

    def _is_point_in_rect(self, point, rect):
        """Controlla se (px, py) è dentro (rx, ry, rw, rh)"""
        px, py = point
        rx, ry, rw, rh = rect
        return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

    def run_analysis(self, img_width=1920, img_height=1080):
        """Esegue il conteggio dei giocatori nelle ROI"""
        # 1. Assicuriamoci di aver caricato le ROI con le dimensioni giuste
        # (Se non abbiamo immagini, usiamo default 1920x1080)
        if not self.rois:
            self._load_config(img_width, img_height)

        # 2. Calcola output path
        # Nome file da specifica PDF: behavior_K_XX.txt
        # Qui usiamo un nome basato sul file di tracking input
        base_name = os.path.basename(self.tracking_path).replace('tracking_', 'behavior_')
        output_txt = os.path.join(self.output_folder, base_name)
        
        print("🧠 Analisi comportamento in corso...")
        
        with open(output_txt, 'w') as f:
            # Ordiniamo i frame per scrivere in sequenza
            sorted_frames = sorted(self.tracks.keys())
            
            for fid in sorted_frames:
                # Inizializza contatori per questo frame
                self.counts[fid] = {'roi1': 0, 'roi2': 0}
                
                players = self.tracks[fid]
                
                # Per ogni ROI
                for roi_name, roi_rect in self.rois.items():
                    count = 0
                    for p in players:
                        if self._is_point_in_rect(p['foot'], roi_rect):
                            count += 1
                            # Segniamo nel dizionario tracks che questo player è in questa roi (utile per video)
                            p[f'in_{roi_name}'] = True
                        else:
                            p[f'in_{roi_name}'] = False
                    
                    self.counts[fid][roi_name] = count
                    
                    # Scrittura su file: frame_id, region_id, n_players
                    # Mappa roi1 -> 1, roi2 -> 2
                    rid = 1 if roi_name == 'roi1' else 2
                    line = f"{fid},{rid},{count}\n"
                    f.write(line)
                    
        print(f"✅ Analisi completata. Risultati: {output_txt}")
        return output_txt

    def create_video_overlay(self, images_folder, output_filename=None):
        """Genera video con ROI evidenziate e contatori"""
        img_files = sorted(glob.glob(os.path.join(images_folder, '*.jpg')))
        if not img_files:
            print("❌ Nessuna immagine trovata per il video behavior.")
            return

        # Setup Video Writer
        first = cv2.imread(img_files[0])
        h, w, _ = first.shape
        
        # Se non abbiamo ancora caricato le ROI, facciamolo ora con le dimensioni reali
        if not self.rois:
            self._load_config(w, h)
            # Rieseguiamo l'analisi se le dimensioni erano diverse dal default
            self.run_analysis(w, h)

        if output_filename is None:
            base = os.path.basename(self.tracking_path).replace('tracking_', 'vis_behavior_').replace('.txt', '.mp4')
            output_filename = os.path.join(self.output_folder, base)

        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
        print(f"🎨 Generazione video behavior: {output_filename}")

        # Colori (BGR)
        colors = {'roi1': (0, 255, 0), 'roi2': (255, 0, 0)} # Verde, Blu
        
        for i, path in enumerate(img_files):
            frame = cv2.imread(path)
            fid = i + 1
            
            # Copia per overlay trasparente
            overlay = frame.copy()
            
            # 1. Disegna le ROI
            for r_name, rect in self.rois.items():
                rx, ry, rw, rh = rect
                c = colors[r_name]
                
                # Rettangolo pieno semitrasparente
                cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), c, -1)
                
                # Bordo solido
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), c, 2)
                
                # Testo Contatore
                count = self.counts.get(fid, {}).get(r_name, 0)
                label = f"{r_name.upper()}: {count}"
                cv2.putText(frame, label, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)

            # Applica trasparenza (0.3 opacity)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # 2. Disegna i Giocatori e il punto piede
            if fid in self.tracks:
                for p in self.tracks[fid]:
                    x, y, bw, bh = p['bbox']
                    fx, fy = p['foot']
                    
                    # Colore dinamico: Giallo se fuori, Verde/Blu se dentro
                    p_color = (0, 255, 255) # Giallo default
                    if p.get('in_roi1'): p_color = colors['roi1']
                    elif p.get('in_roi2'): p_color = colors['roi2']
                    
                    # Box leggero
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), p_color, 1)
                    
                    # PALLINO SUL PIEDE (Fondamentale per debuggare)
                    cv2.circle(frame, (fx, fy), 6, p_color, -1) # Pieno
                    cv2.circle(frame, (fx, fy), 6, (0,0,0), 1)  # Bordo nero

            out.write(frame)
            if i % 100 == 0: print(f"Rendering {i}/{len(img_files)}...")

        out.release()
        print("✅ Video behavior salvato.")