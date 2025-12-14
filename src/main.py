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
        python src/main.py --config config/main_config.yaml --step all --seq SNMOT-060 SNMOT-061
    Solo tracking:
        python src/main.py --config config/main_config.yaml --step tracking --seq SNMOT-060
    Behaiour e visualizer:
        python src/main.py --config config/main_config.yaml --step behaviour visualizer --seq SNMOT-060
    """
    parser = argparse.ArgumentParser(description="SoccerNet Pipeline")
    parser.add_argument('--config', type=str, default='configs/main_config.yaml', help='Path al file config')
    parser.add_argument('--step', type=str, nargs='+', default=['all'], 
                        choices=['tracking', 'behaviour', 'visualizer', 'eval', 'all'], 
                        help='Step da eseguire. Puoi indicarne più di uno (es: --step behaviour visualizer)')
    parser.add_argument('--seq', type=str, nargs='+', help='Lista sequenze (es. SNMOT-060).')

    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # Determina sequenze
    if args.seq:
        sequences = args.seq
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
        print("\n📊 Avvio Valutazione (HOTA)...")
        evaluator = Evaluator(cfg)
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