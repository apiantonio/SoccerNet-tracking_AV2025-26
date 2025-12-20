import os
import sys

def rename_soccernet_folders(root_dir, revert=False):
    """
    Rinomina le cartelle dataset SoccerNet:
    - Default: da "SNMOT-XYZ" a "XYZ"
    - Revert=True: da "XYZ" a "SNMOT-XYZ" (per tornare indietro se serve)
    """
    
    if not os.path.exists(root_dir):
        print(f"Errore: La directory '{root_dir}' non esiste.")
        return

    count = 0
    # topdown=False è importante per rinominare le directory partendo dalle più profonde
    # (anche se in questo caso le cartelle SNMOT sono solitamente foglie o quasi)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            
            should_rename = False
            new_dirname = ""

            if not revert:
                # Caso: Rimuovere SNMOT-
                if dirname.startswith("SNMOT-"):
                    new_dirname = dirname.replace("SNMOT-", "")
                    should_rename = True
            else:
                # Caso: Aggiungere SNMOT- (Revert)
                # Controlliamo se è solo numerico (es. "060") o alfanumerico breve
                # per evitare di rinominare cartelle di sistema tipo "img1" o "gt"
                if not dirname.startswith("SNMOT-") and dirname.replace(".", "").isdigit():
                    # Nota: isdigit() controlla se è un numero (es. 060). 
                    # Se le tue cartelle contengono lettere ma non SNMOT, togli il check isdigit.
                    new_dirname = f"SNMOT-{dirname}"
                    should_rename = True

            if should_rename:
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)

                if os.path.exists(new_path):
                    print(f"[SKIP] {dirname} -> {new_dirname}: La destinazione esiste già.")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"[OK] {dirname} -> {new_dirname}")
                        count += 1
                    except Exception as e:
                        print(f"[ERR] Impossibile rinominare {dirname}: {e}")

    action = "ripristinate" if revert else "rinominate"
    print(f"\nOperazione completata. {count} cartelle {action}.")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    # Inserisci qui il percorso della tua cartella tracking principale
    # Può essere "./tracking" o "./tracking/test" ecc.
    TARGET_DIR = "./tracking" 
    
    # Vuoi tornare indietro? Metti True
    REVERT_MODE = False 
    # ----------------------

    print(f"Avvio rinomina in: {os.path.abspath(TARGET_DIR)}")
    print("Premi INVIO per continuare o CTRL+C per annullare...")
    try:
        input()
    except KeyboardInterrupt:
        sys.exit()

    rename_soccernet_folders(TARGET_DIR, revert=REVERT_MODE)