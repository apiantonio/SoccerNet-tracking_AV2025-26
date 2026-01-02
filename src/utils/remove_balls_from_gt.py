import os
import configparser
import shutil

def remove_balls_from_gt(dataset_root):
    """
    Rimuove le righe corrispondenti alla palla ('ball') dai file gt.txt
    basandosi sul mapping presente in gameinfo.ini.
    Crea un backup del file originale come gt.txt.bak.
    """
    
    if not os.path.exists(dataset_root):
        print(f"Errore: La directory '{dataset_root}' non esiste.")
        return

    sequences_processed = 0
    total_balls_removed = 0

    # Scansiona le cartelle (es. test/SNMOT-060 o test/060)
    for seq_name in os.listdir(dataset_root):
        seq_path = os.path.join(dataset_root, seq_name)
        
        if not os.path.isdir(seq_path):
            continue

        gameinfo_path = os.path.join(seq_path, "gameinfo.ini")
        gt_file_path = os.path.join(seq_path, "gt", "gt.txt")

        # Verifica che esistano sia gameinfo che gt
        if not os.path.exists(gameinfo_path):
            # Alcune cartelle potrebbero non avere gameinfo (es. se non sono sequenze raw)
            continue
        
        if not os.path.exists(gt_file_path):
            print(f"[WARN] GT mancante in {seq_name}")
            continue

        # --- STEP 1: Trova gli ID della palla in gameinfo.ini ---
        ball_ids = []
        try:
            config = configparser.ConfigParser()
            config.read(gameinfo_path)
            
            if 'Sequence' in config:
                section = config['Sequence']
                for key, value in section.items():
                    # Le chiavi in configparser sono lowercase per default (trackletid_x)
                    if key.startswith("trackletid_"):
                        # Controlla se il valore contiene "ball" (es. "ball;1" o "ball;none")
                        if "ball" in value.lower():
                            # Estrae l'ID numerico dalla chiave (es. trackletid_6 -> 6)
                            try:
                                track_id = int(key.split('_')[1])
                                ball_ids.append(track_id)
                            except ValueError:
                                pass
        except Exception as e:
            print(f"[ERR] Errore lettura gameinfo in {seq_name}: {e}")
            continue

        if not ball_ids:
            print(f"[INFO] Nessuna palla trovata nei metadati di {seq_name}. Skip.")
            continue

        # --- STEP 2: Filtra il file GT ---
        # Crea backup se non esiste
        backup_path = gt_file_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy(gt_file_path, backup_path)

        lines_kept = []
        balls_in_seq = 0
        
        with open(gt_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            
            try:
                # La colonna 1 (indice 1) è il Track ID
                # Format: frame, ID, x, y, w, h, conf, -1, -1, -1
                current_id = int(parts[1])
                
                if current_id in ball_ids:
                    balls_in_seq += 1
                else:
                    lines_kept.append(line)
            except ValueError:
                # Se c'è qualche riga corrotta, la manteniamo per sicurezza
                lines_kept.append(line)

        # --- STEP 3: Scrivi il nuovo GT ---
        if balls_in_seq > 0:
            with open(gt_file_path, 'w') as f:
                f.writelines(lines_kept)
            print(f"[OK] {seq_name}: Rimossi {balls_in_seq} record palla (IDs: {ball_ids})")
            total_balls_removed += balls_in_seq
            sequences_processed += 1
        else:
            print(f"[SKIP] {seq_name}: ID palla trovati {ball_ids} ma nessuna riga nel GT corrisponde.")

    print(f"\nOperazione completata.")
    print(f"Sequenze modificate: {sequences_processed}")
    print(f"Totale annotazioni palla rimosse: {total_balls_removed}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    # Percorso della cartella contenente le sequenze (es. ./tracking/test)
    # Assicurati che punti alla cartella che contiene le sottocartelle (060, 061...)
    DATASET_PATH = "SIMULATOR/lecture_example_from_training/test_set/videos"
    # ----------------------

    print(f"Avvio rimozione palle dai GT in: {os.path.abspath(DATASET_PATH)}")
    print("Verranno creati backup .bak dei file gt.txt originali.")
    remove_balls_from_gt(DATASET_PATH)