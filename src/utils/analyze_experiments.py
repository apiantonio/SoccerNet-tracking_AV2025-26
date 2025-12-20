import os
import json
import argparse
import pandas as pd
import glob

def load_experiments(output_folder):
    """
    Scansiona ricorsivamente la cartella output e legge tutti i file JSON.
    Estrae metriche, numero di sequenze e parametri di configurazione chiave.
    """
    experiments = []
    
    # Cerca tutti i file .json ricorsivamente nella cartella output
    search_pattern = os.path.join(output_folder, "**", "*.json")
    files = glob.glob(search_pattern, recursive=True)

    print(f"🔍 Trovati {len(files)} file JSON in '{output_folder}'. Elaborazione in corso...")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Verifica se il JSON ha la struttura attesa (contiene 'metrics_overall')
            if "metrics_overall" not in data:
                continue

            # --- 1. Estrazione Metriche ---
            metrics = data.get("metrics_overall", {})
            
            # --- 2. Conteggio Sequenze ---
            # Contiamo le sequenze effettivamente valutate
            processed_seqs = data.get("metrics_per_sequence", [])
            num_seqs = len(processed_seqs)
            
            # Fallback: se metrics_per_sequence è vuoto, proviamo a leggere dal config
            if num_seqs == 0:
                settings = data.get("main_config", {}).get("settings", {})
                seq_list = settings.get("sequences", [])
                if isinstance(seq_list, list):
                    num_seqs = len(seq_list)
                elif seq_list == "all":
                    num_seqs = "ALL" # Placeholder se non sappiamo il numero esatto
            
            # --- 3. Estrazione Parametri Chiave (Flattening) ---
            main_conf = data.get("main_config", {})
            tracker_settings = main_conf.get("tracker", {})
            algo_conf = data.get("tracker_config", {})
            
            # Costruiamo il dizionario per il DataFrame
            entry = {
                # Info File
                "Folder": os.path.basename(os.path.dirname(file_path)),
                "Filename": os.path.basename(file_path),
                "Path": file_path,
                
                # Numero Sequenze
                "# Seq": num_seqs,

                # Metriche Principali
                "PTBS": metrics.get("PTBS", "N/A"),
                "HOTA_05": metrics.get("HOTA_05", "N/A"),
                "nMAE": metrics.get("nMAE", "N/A"),
                "AssA": metrics.get("AssA", "N/A"),
                "DetA": metrics.get("DetA", "N/A"),
                "TP": metrics.get("counts_sum", {}).get("TP", "N/A"),
                "FP": metrics.get("counts_sum", {}).get("FP", "N/A"),
                "FN": metrics.get("counts_sum", {}).get("FN", "N/A"),
                "TPA": metrics.get("counts_sum", {}).get("TPA", "N/A"),
                "FNA": metrics.get("counts_sum", {}).get("FNA", "N/A"),
                "FPA": metrics.get("counts_sum", {}).get("FPA", "N/A"),
                
                # Parametri Importanti
                "Conf": tracker_settings.get("conf", "N/A"),
                "IoU": tracker_settings.get("iou", "N/A"),
                "ImgSz": tracker_settings.get("imgsz", "N/A"),
                "Buffer": algo_conf.get("track_buffer", "N/A"),
                "High_Th": algo_conf.get("track_high_thresh", "N/A"),
                "Low_Th": algo_conf.get("track_low_thresh", "N/A"),
                "New_Th": algo_conf.get("new_track_thresh", "N/A"),
                "Match_Th": algo_conf.get("match_thresh", "N/A"),
                "Fuse": algo_conf.get("fuse_score", "N/A"),
                "ReID": algo_conf.get("with_reid", False),
                "App_Th": algo_conf.get("appearance_thresh", "N/A") if algo_conf.get("with_reid", False) else "-",
                "Prox_Th": algo_conf.get("proximity_thresh", "N/A") if algo_conf.get("with_reid", False) else "-",
                "ReID_Model": algo_conf.get("reid_model", "N/A") if algo_conf.get("with_reid", False) else "-"
            }
            
            experiments.append(entry)
            
        except Exception as e:
            print(f"⚠️ Errore lettura {file_path}: {e}")

    return pd.DataFrame(experiments)

def main():
    parser = argparse.ArgumentParser(description="Analizza e classifica gli esperimenti SoccerNet.")
    
    parser.add_argument("--folder", type=str, default="output", 
                        help="Cartella radice dove cercare i JSON (default: output)")
    
    parser.add_argument("--sort", type=str, nargs="+", default=["PTBS"], 
                        help="Metrica/e per l'ordinamento (es. PTBS HOTA_05 nMAE). Default: PTBS")
    
    parser.add_argument("--top", type=int, default=5, 
                        help="Numero di risultati migliori da mostrare (default: 5)")
    
    parser.add_argument("--minseq", type=int, default=0,
                        help="Filtra esperimenti che hanno meno di N sequenze (default: 0 = mostra tutti)")
    
    parser.add_argument("--csv", type=str, default="output/report.csv",
                        help="Nome file opzionale per salvare il report CSV")

    args = parser.parse_args()

    # 1. Carica Dati
    df = load_experiments(args.folder)

    if df.empty:
        print("❌ Nessun esperimento valido trovato.")
        return
    
    if args.minseq > 0:
        # Convertiamo la colonna '# Seq' in numerico (gestendo "ALL" o errori come NaN -> 0)
        # Questo permette di filtrare via i file incompleti o corrotti
        temp_seq_col = pd.to_numeric(df['# Seq'], errors='coerce').fillna(0)
        
        # Filtriamo il dataframe originale
        df_filtered = df[temp_seq_col >= args.minseq]
        
        if df_filtered.empty:
            print(f"❌ Nessun esperimento trovato con almeno {args.minseq} sequenze.")
            print(f"(Trovati {len(df)} esperimenti totali, ma tutti con meno sequenze)")
            return

        print(f"ℹ️ Filtrati {len(df) - len(df_filtered)} esperimenti con meno di {args.minseq} sequenze.")
        df = df_filtered

    # 2. Controllo colonne per ordinamento
    sort_cols = args.sort
    valid_sort_cols = [c for c in sort_cols if c in df.columns]
    
    if not valid_sort_cols:
        print(f"⚠️ Le metriche specificate {sort_cols} non esistono. Uso PTBS di default.")
        valid_sort_cols = ["PTBS"]

    # 3. Ordinamento
    df_sorted = df.sort_values(by=valid_sort_cols, ascending=False)

    # 4. Selezione Top N
    top_df = df_sorted.head(args.top)

    # 5. Visualizzazione
    print(f"\n🏆 TOP {args.top} ESPERIMENTI (Ordinati per: {valid_sort_cols})")
    print("=" * 160)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    cols_to_show = ["Filename", "# Seq", "PTBS", "HOTA_05", "nMAE", "DetA", "AssA", "Imgsz", "Conf", "IoU", "Buffer", "High_Th", "Low_Th", "New_Th", "Match_Th", "ReID"]
    
    # Filtriamo per mostrare solo le colonne che esistono davvero nel DF (sicurezza)
    existing_cols = [c for c in cols_to_show if c in top_df.columns]
    
    print(top_df[existing_cols].to_string(index=False))
    
    print("=" * 160)

    # 6. Export CSV
    if args.csv:
        # Assicuriamoci che la cartella di destinazione esista
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        df_sorted.to_csv(args.csv, index=False)
        print(f"💾 Report completo salvato in: {args.csv}")

if __name__ == "__main__":
    main()