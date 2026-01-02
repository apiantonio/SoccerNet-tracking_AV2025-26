import argparse
import yaml
import os
import time
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
    if args.seq:
        cfg['settings']['sequences'] = args.seq
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
    if args.field_mask:
        if 'True' in args.field_mask:
            cfg['tracker']['field_mask'] = True
        elif 'False' in args.field_mask:
            cfg['tracker']['field_mask'] = False

    if 'none' in args.debug:
        cfg['debug'] = False
    else:
        # Crea un dizionario di configurazione per il debug
        cfg['debug'] = {
            'show_track': 'show_tracks' in args.debug,
            'show_mask': 'show_mask' in args.debug,
            'show_behaviour': 'show_behaviour' in args.debug
        }

    return cfg


def format_time(seconds):
    """Formatta i secondi in un formato leggibile (ore, minuti e secondi)."""
    if seconds is None or seconds < 0:
        return "N/A"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"
    
def main():
    """Funzione principale per l'esecuzione della pipeline di analisi video SoccerNet.

    Questa funzione orchestra le diverse fasi della pipeline: tracking, analisi comportamentale,
    valutazione e visualizzazione. È possibile eseguire l'intera pipeline o solo fasi specifiche
    utilizzando gli argomenti da linea di comando.

    Fasi (step) disponibili:
        - tracking: Esegue il tracciamento degli oggetti (calciatori, pallone) nei video.
        - behaviour: Analizza i tracciamenti per estrarre comportamenti specifici.
        - visualizer: Genera video con le informazioni di tracking e analisi sovrapposte.
        - eval: Esegue la valutazione delle performance del tracking (es. HOTA).
        - all: Esegue tutte le fasi in sequenza (impostazione predefinita).

    Argomenti da linea di comando:
        --config (-c):
            Path al file di configurazione principale (YAML). Default: 'configs/main_config.yaml'.
        --step (-s):
            Una o più fasi della pipeline da eseguire. Default: ['all'].
        --seq:
            Lista di sequenze da processare (es. 'SNMOT-060'), o 'all' per tutte le sequenze
            nella cartella di input. Default: ['all'].
        --input_folder (-i):
            (Opzionale) Sovrascrive il percorso della cartella di input specificato nel file di configurazione.
        --output_folder (-o):
            (Opzionale) Sovrascrive il percorso della cartella di output specificato nel file di configurazione.
        --tracker_config:
            (Opzionale) Sovrascrive il path al file di configurazione del tracker.
        --roi_config:
            (Opzionale) Sovrascrive il path al file di configurazione delle ROI (Region of Interest).
        --imgsz:
            (Opzionale) Sovrascrive la risoluzione di input per il modello di tracking.
        --conf:
            (Opzionale) Sovrascrive la soglia di confidenza minima per il tracking.
        --iou:
            (Opzionale) Sovrascrive la soglia di IOU (Intersection over Union) per il NMS.
        --batch:
            (Opzionale) Sovrascrive la dimensione del batch per l'inferenza del tracker.
        --half (--fp16, -hp):
            (Opzionale) Abilita l'inferenza a precisione ridotta (FP16) se supportata.
        --field_mask:
            (Opzionale) Specifica se utilizzare la maschera del campo per filtrare le detection.
            Valori possibili: 'True' o 'False'. Default: ['True'].
        --verbose (-v):
            (Opzionale) Abilita un output più dettagliato (verboso) durante il tracking.
        --debug:
            (Opzionale) Abilita modalità di debug visuale. Valori possibili:
            'show_tracks', 'show_mask', 'show_behaviour'. 'none' per disabilitare. Default: ['none'].

    Esempio di utilizzo:
        Eseguire l'intera pipeline su due sequenze specifiche, sovrascrivendo le cartelle di input/output:
        ```
        python src/main.py --config configs/main_config.yaml --step all --seq 060 061 --input_folder data/input --output_folder data/output
        ```
        Eseguire solo la fase di tracking con una confidenza personalizzata:
        ```
        python src/main.py --step tracking --seq 060 --conf 0.35
        ```
    """
    parser = argparse.ArgumentParser(description="SoccerNet Pipeline")
    parser.add_argument('-c', '--config', type=str, default='configs/main_config.yaml', help='Path al file config')
    parser.add_argument('-s', '--step', type=str, nargs='+', default=['all'],
                        choices=['tracking', 'behaviour', 'eval', 'visualizer', 'all'],
                        help='Step da eseguire. Puoi indicarne più di uno (es: --step behaviour visualizer)')
    parser.add_argument('--seq', type=str, nargs='+',
                        help='Lista sequenze (es. SNMOT-060), o "all" per tutte le sequenze nel folder di input',
                        default=['all'])

    # Argomenti opzionali per sovrascrivere i path nel file di config
    parser.add_argument('-i', '--input_folder', type=str, help='Sovrascrivi cartella input dal config')
    parser.add_argument('-o', '--output_folder', type=str, help='Sovrascrivi cartella output dal config')
    parser.add_argument('--tracker_config', type=str,
                        help='Path al file di configurazione del tracker (opzionale sovrascrittura)')
    parser.add_argument('--roi_config', type=str,
                        help='Path al file di configurazione delle ROI (opzionale sovrascrittura)')

    parser.add_argument('--imgsz', type=int,
                        help='Sovrascrivi la risoluzione di input del tracker (es. 640, 960, 1088, 1280)')
    parser.add_argument('--conf', type=float, help='Sovrascrivi la confidenza minima del tracker (es. 0.25)')
    parser.add_argument('--iou', type=float, help='Sovrascrivi la soglia IOU del tracker (es. 0.7)')
    parser.add_argument('--batch', type=int, help='Sovrascrivi la dimensione del batch per il tracker (se supportato)')
    parser.add_argument('-hp', '--fp16', '--half', dest='half', help='Usa FP16 per il tracking (se supportato)',
                        action='store_true', default=None)
    parser.add_argument('--field_mask', type=str, nargs='+', choices=['True', 'False'], default=['True'],
                        help='Usa la maschera del campo per filtrare le detections fuori campo (True/False)')

    parser.add_argument('-v', '--verbose', action='store_true', help='Abilita output verboso per il tracker',
                        default=False)
    parser.add_argument('--debug', type=str, nargs='+', choices=['show_tracks', 'show_mask', 'show_behaviour', 'none'],
                        default=['none'],
                        help='Modalità debug: show_tracks, show_mask, show_behaviour o none per disabilitare')

    args = parser.parse_args()  # Parsing argomenti da linea di comando
    cfg = load_config(args.config)  # Caricamento config principale

    cfg = finalize_arguments(args, cfg)  # Sovrascrivi config con argomenti da linea di comando se presenti

    if args.seq:
        sequences = args.seq
        if sequences == ['all']:
            input_root = cfg['paths']['input_folder']
            sequences = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    elif cfg['settings'].get('sequences'):  # se sequence non è definito negli argomenti, controlla nel file di config
        sequences = cfg['settings']['sequences']
    else:
        input_root = cfg['paths']['input_folder']
        sequences = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    print(f"Configurazione caricata. Step: {args.step}")
    print(f"Sequenze target: {sequences}")

    start_time = time.time()

    # --- FASE 1: TRACKING ---
    start_tracking_time = time.time()
    if 'all' in args.step or 'tracking' in args.step:
        tracker = SoccerTracker(cfg)
        num_sequences = len(sequences)
        for i, seq in enumerate(sequences):
            try:
                print(f"\n[{i + 1}/{num_sequences}] Avvio tracking su '{seq}'...")
                start_track = time.time()
                tracker.track_sequence(seq)
                curr_time = time.time()

                elapsed_time = curr_time - start_tracking_time
                progress = (i + 1) / num_sequences
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time

                print(f"[{i + 1}/{num_sequences}] Fine tracking su '{seq}' in {format_time(curr_time - start_track):>7} | "
                      f"tascorso/totale: {format_time(elapsed_time):>9} / {format_time(estimated_total_time):<9} circa | "
                      f"rimanente: {format_time(remaining_time):<9} circa |")

            except Exception as e:
                print(f"Errore tracking su {seq}: {e}")
        end_tracking_time = time.time()
        tracking_time = end_tracking_time - start_tracking_time
        if sequences:
            mean_tracking_time = tracking_time / len(sequences)
            print(f"> Tempo totale di tracking: {format_time(tracking_time)} (media: {mean_tracking_time:.2f} secondi/sequenza)")
        else:
            print(f"> Tempo totale di tracking: {format_time(tracking_time)}")


    # --- FASE 2: BEHAVIOUR ---
    start_behaviour_time = time.time()
    if 'all' in args.step or 'behaviour' in args.step:
        analyzer = BehaviorAnalyzer(cfg)
        num_sequences = len(sequences)
        for i, seq in enumerate(sequences):
            try:
                print(f"\n[{i + 1}/{num_sequences}] Avvio analisi comportamento su '{seq}'...")
                analyzer.process_sequence(seq)
            except Exception as e:
                print(f"Errore behaviour su {seq}: {e}")
        end_behaviour_time = time.time()
        behaviour_time = end_behaviour_time - start_behaviour_time
        if sequences:
            mean_behaviour_time = behaviour_time / len(sequences)
            print(f"> Tempo totale di behaviour analysis: {format_time(behaviour_time)} (media: {mean_behaviour_time:.2f} secondi/sequenza)")
        else:
            print(f"> Tempo totale di behaviour analysis: {format_time(behaviour_time)}")


    # --- FASE 3: EVALUATION ---
    start_evaluation_time = time.time()
    if 'all' in args.step or 'eval' in args.step:
        evaluator = Evaluator(cfg)
        print("\nAvvio Valutazione...")
        try:
            evaluator.evaluate(sequences)
        except Exception as e:
            print(f"Errore durante la valutazione: {e}")
        end_evaluation_time = time.time()
        evaluation_time = end_evaluation_time - start_evaluation_time
        print(f"> Tempo totale di valutazione: {format_time(evaluation_time)}")

    print(f"\n> Pipeline completata in {format_time(time.time() - start_time)}.\n")

    # --- FASE 4: VISUALIZER ---
    start_video_time = time.time()
    if 'all' in args.step or 'visualizer' in args.step:
        print("\nAvvio generazione video...")
        vis = SoccerVisualizer(cfg)
        num_sequences = len(sequences)
        for i, seq in enumerate(sequences):
            try:
                print(f"\n[{i + 1}/{num_sequences}] Avvio generazione video per '{seq}'...")
                vis.generate_video(seq)
            except Exception as e:
                print(f"Errore visualizer su {seq}: {e}")
        end_video_time = time.time()
        video_time = end_video_time - start_video_time
        if sequences:
            mean_video_time = video_time / len(sequences)
            print(f"\nGenerazione video completata in {format_time(video_time)} (media: {mean_video_time:.2f} secondi/sequenza).")
        else:
            print(f"\nGenerazione video completata in {format_time(video_time)}.")


if __name__ == "__main__":
    main()