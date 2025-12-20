import os
import sys

def rename_prediction_files(target_dir):
    """
    Scansiona ricorsivamente target_dir e rinomina i file delle predizioni:
    Da: tracking_SNMOT-123_16.txt  -> tracking_123_16.txt
    Da: behavior_SNMOT-123_16.txt  -> behavior_123_16.txt
    """
    
    if not os.path.exists(target_dir):
        print(f"Errore: La directory '{target_dir}' non esiste.")
        return

    count = 0
    
    # os.walk permette di scendere in tutte le sottocartelle
    for dirpath, _, filenames in os.walk(target_dir):
        for filename in filenames:
            
            # Controlliamo se è un file txt e se contiene il prefisso incriminato
            if filename.endswith(".txt") and "SNMOT-" in filename:
                
                # Semplice sostituzione della stringa
                new_filename = filename.replace("SNMOT-", "")
                
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                
                # Evitiamo di sovrascrivere se il file esiste già
                if os.path.exists(new_path):
                    print(f"[SKIP] {filename} -> {new_filename}: Il file di destinazione esiste già.")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"[OK] {filename} -> {new_filename}")
                        count += 1
                    except Exception as e:
                        print(f"[ERR] Errore rinominando {filename}: {e}")

    print(f"\nOperazione completata. {count} file rinominati.")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    # Inserisci qui il percorso della cartella output o dell'esperimento specifico
    # Esempio: "./output/test_all_conf00001_iou60_botsort7"
    TARGET_DIR = "./output/test_botsortprova1"
    # ----------------------

    print(f"Avvio rinomina predizioni in: {os.path.abspath(TARGET_DIR)}")
    print("Verrà rimosso 'SNMOT-' da tutti i file .txt trovati.")
    print("Premi INVIO per continuare...")
    try:
        input()
    except KeyboardInterrupt:
        sys.exit()

    rename_prediction_files(TARGET_DIR)