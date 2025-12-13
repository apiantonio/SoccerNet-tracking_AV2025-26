import trackeval
import numpy as np
import os
import csv


class SoccerNet6ColDataset(trackeval.datasets.MotChallenge2DBox):
    def _load_raw_file(self, tracker, seq, is_gt):
        """
        Questa funzione sovrascrive quella originale.
        Legge file a 6 colonne e aggiunge le 4 colonne mancanti in memoria.
        """
        # Determina il percorso del file
        if is_gt:
            file = os.path.join(self.gt_folder, seq, 'gt', 'gt.txt')
        else:
            file = os.path.join(self.tracker_folder, tracker, self.tracker_sub_folder, seq + '.txt')

        # Se il file non esiste, ritorna vuoto
        if not os.path.isfile(file):
            return np.empty((0, 10))  # Ritorna vuoto ma con 10 colonne (standard interno)

        # Leggi i dati grezzi
        raw_data = []
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # Salta linee vuote
                if not row: continue

                # Converti in float
                vals = [float(x) for x in row]

                # SE SONO 6 COLONNE (il tuo formato):
                if len(vals) == 6:
                    # Aggiungiamo: conf=1, x=-1, y=-1, z=-1
                    vals.extend([1.0, -1.0, -1.0, -1.0])

                raw_data.append(vals)

        if len(raw_data) == 0:
            return np.empty((0, 10))

        return np.array(raw_data)


# ---------------------------------------------------------
# 2. Configurazione e Calcolo (Uguale a prima ma usa la nuova classe)
# ---------------------------------------------------------

# Aggiorna i percorsi
GT_FOLDER = './path/to/your/gt_folder'
TRACKERS_FOLDER = './path/to/your/results'

eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config['DISPLAY_LESS_PROGRESS'] = True
eval_config['PRINT_RESULTS'] = False
eval_config['PRINT_ONLY_COMBINED'] = True

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config['GT_FOLDER'] = GT_FOLDER
dataset_config['TRACKERS_FOLDER'] = TRACKERS_FOLDER

# ATTENZIONE: Qui usiamo la NOSTRA classe personalizzata invece di MotChallenge2DBox
dataset_list = [SoccerNet6ColDataset(dataset_config)]

metrics_config = {'METRICS': ['HOTA']}
evaluator = trackeval.Evaluator(eval_config)
metrics_list = [trackeval.metrics.HOTA(metrics_config)]

# Esegui valutazione
raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

# Estrai HOTA(0.5) come visto prima
dataset_name = dataset_list[0].get_name()
tracker_list = list(raw_results[dataset_name].keys())
tracker_name = tracker_list[0]
hota_results = raw_results[dataset_name][tracker_name]['COMBINED_SEQ']['HOTA']
hota_05_score = hota_results['HOTA_alphas'][9]

print(f"--------------------------------------------------")
print(f"HOTA(0.5) calcolato direttamente dai file a 6 colonne:")
print(f"{hota_05_score * 100:.2f}%")
print(f"--------------------------------------------------")