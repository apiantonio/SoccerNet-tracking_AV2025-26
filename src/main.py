import argparse
import yaml
import os
import sys

# Import classi
from tracker.soccer_tracker import SoccerTracker
from behaviour.behaviour_analyzer import BehaviorAnalyzer
from visualizer.soccer_visualizer import SoccerVisualizer
from evaluation.evaluator import Evaluator 

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def finalize_arguments(args, cfg):
    """Sovrascrivi gli argomenti nel file di config con quelli passati da linea di comando (se presenti)"""
    if args.tracker_config:
        cfg['paths']['tracker_config'] = args.tracker_config 
    if args.input_folder:
        cfg['paths']['input_folder'] = args.input_folder
    if args.output_folder:
        cfg['paths']['output_folder'] = args.output_folder
    if args.roi_config:
        cfg['paths']['roi_config'] = args.roi_config
    if args.imgsz:
        cfg['tracker']['imgsz'] = args.imgsz
    if args.half is not None:
        cfg['tracker']['half'] = args.half
    if args.verbose:
        cfg['tracker']['verbose'] = True
    if args.conf:
        cfg['tracker']['conf'] = args.conf
    if args.iou:
        cfg['tracker']['iou'] = args.iou
    if args.batch:
        cfg['tracker']['batch'] = args.batch
    if 'none' in args.debug:
        cfg['debug'] = False
    else:
        # Crea un dizionario di configurazione per il debug
        cfg['debug'] = {
            'show_track': 'show_tracks' in args.debug,
            'show_behaviour': 'show_behaviour' in args.debug
        }
        
    return cfg

def main():
    """
    Main function per eseguire la pipeline SoccerNet.
    
    Opzioni step:
        - tracking: esegue solo il tracking
        - behaviour: esegue solo l'analisi del comportamento
        - visualizer: genera solo i video visualizzati
        - eval: esegue solo la valutazione (HOTA)
        - all: esegue tutta la pipeline (default)
        
    Esempio di esecuzione:
        python src/main.py --config configs/main_config.yaml --step all --seq SNMOT-060 SNMOT-061 --input_folder data/input --output_folder data/output
    """
    parser = argparse.ArgumentParser(description="SoccerNet Pipeline")
    parser.add_argument('-c', '--config', type=str, default='configs/main_config.yaml', help='Path al file config')
    parser.add_argument('-s','--step', type=str, nargs='+', default=['all'], 
                        choices=['tracking', 'behaviour', 'eval', 'visualizer', 'all'], 
                        help='Step da eseguire. Puoi indicarne più di uno (es: --step behaviour visualizer)')
    parser.add_argument('--seq', type=str, nargs='+', help='Lista sequenze (es. SNMOT-060), o "all" per tutte le sequenze nel folder di input', default=['all'])
    
    # Argomenti opzionali per sovrascrivere i path nel file di config
    parser.add_argument('-i', '--input_folder', type=str, help='Sovrascrivi cartella input dal config')
    parser.add_argument('-o', '--output_folder', type=str, help='Sovrascrivi cartella output dal config')
    parser.add_argument('--tracker_config', type=str, help='Path al file di configurazione del tracker (opzionale sovrascrittura)')
    parser.add_argument('--roi_config', type=str, help='Path al file di configurazione delle ROI (opzionale sovrascrittura)')
    
    parser.add_argument('--imgsz', type=int, help='Sovrascrivi la risoluzione di input del tracker (es. 640, 960, 1088, 1280)')
    parser.add_argument('--conf', type=float, help='Sovrascrivi la confidenza minima del tracker (es. 0.25)')
    parser.add_argument('--iou', type=float, help='Sovrascrivi la soglia IOU del tracker (es. 0.7)')
    parser.add_argument('--batch', type=int, help='Sovrascrivi la dimensione del batch per il tracker (se supportato)')
    parser.add_argument('--show_video', action='store_true', help='Mostra il video di tracking in tempo reale durante l\'esecuzione', default=False)
    parser.add_argument('-hp', '--fp16', '--half', dest='half', help='Usa FP16 per il tracking (se supportato)', action='store_true', default=None)
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Abilita output verboso per il tracker', default=False)
    parser.add_argument('--debug', type=str, nargs='+', choices=['show_tracks', 'show_behaviour', 'none'], default=['none'],
                        help='Modalità debug: show_tracks, show_behaviour o entrambi')
     
    args = parser.parse_args()     # Parsing argomenti da linea di comando
    cfg = load_config(args.config) # Caricamento config principale
    
    cfg = finalize_arguments(args, cfg) # Sovrascrivi config con argomenti da linea di comando se presenti
    
    if args.seq:
        sequences = args.seq
        if sequences == ['all']:
            input_root = cfg['paths']['input_folder']
            sequences = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
            sequences = [s for s in sequences if s.startswith("SNMOT")]
    elif cfg['settings'].get('sequences'): # se sequence non è definito negli argomenti, controlla nel file di config
        sequences = cfg['settings']['sequences']
    else:
        input_root = cfg['paths']['input_folder']
        sequences = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
        sequences = [s for s in sequences if s.startswith("SNMOT")]

    print(f"📋 Configurazione caricata. Step: {args.step}")
    print(f"📂 Sequenze target: {sequences}")

    # --- FASE 1: TRACKING ---
    if 'all' in args.step or 'tracking' in args.step:
        tracker = SoccerTracker(cfg)
        for seq in sequences:
            try:
                tracker.track_sequence(seq)
            except Exception as e:
                print(f"❌ Errore tracking su {seq}: {e}")

    # --- FASE 2: BEHAVIOUR ---
    if 'all' in args.step or 'behaviour' in args.step:
        analyzer = BehaviorAnalyzer(cfg)
        for seq in sequences:
            try:
                analyzer.process_sequence(seq)
            except Exception as e:
                print(f"❌ Errore behaviour su {seq}: {e}")

    # --- FASE 3: EVALUATION ---
    if 'all' in args.step or 'eval' in args.step:
        evaluator = Evaluator(cfg)
        print("\n📊 Avvio Valutazione...")
        try:
            evaluator.evaluate(sequences)
        except Exception as e:
            print(f"❌ Errore durante la valutazione: {e}")

    # --- FASE 4: VISUALIZER ---
    if 'all' in args.step or 'visualizer' in args.step:
        print("\n🎨 Avvio generazione video...")
        vis = SoccerVisualizer(cfg)
        for seq in sequences:
            try:
                vis.generate_video(seq)
            except Exception as e:
                print(f"❌ Errore visualizer su {seq}: {e}")

if __name__ == "__main__":
    main()