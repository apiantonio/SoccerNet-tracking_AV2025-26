
# ⚽ SoccerNet: Player Detection, Tracking and Behavior Analysis

**Artificial Vision Project Work 2024/2025** *University of Salerno - Dept. of Information Engineering, Electrical Engineering and Applied Mathematics*

**Autori:** Antonio Apicella, Antonio Graziosi  
**Gruppo:** 16

---

## 📖 Introduzione

Questo progetto presenta una pipeline completa di Computer Vision sviluppata per la **SoccerNet Video Understanding Benchmark Suite**. L'obiettivo è affrontare due task principali su clip video di partite di calcio reali:

1.  **Player Detection & Tracking:** Rilevare e tracciare univocamente i giocatori e gli arbitri in campo ("persone di interesse"), mantenendo l'identità consistente nonostante occlusioni, movimenti rapidi della telecamera e l'assenza di caratteristiche visive distintive (divise identiche).
2.  **Behavior Analysis:** Stimare la densità dei giocatori in specifiche *Region of Interest* (ROI) del campo per ogni frame.

La soluzione proposta si basa su un approccio **Tracking-by-Detection** ottimizzato, che combina un detector allo stato dell'arte (**YOLOv11x**) configurato per l'alta sensibilità (*High Recall*), un modulo di **Field Masking** adattivo per la rimozione dei falsi positivi a bordo campo, e un tracker basato sul movimento (**BoT-SORT** con compensazione globale della telecamera GMC), privo di moduli di Re-Identificazione visiva.

Il sistema è stato valutato sul Test Set della challenge, raggiungendo un punteggio PTBS (*Player Tracking and Behavior Score*) di **1.527**, con un HOTA di **0.742**.

---

## 🚀 Caratteristiche Principali

* **Detector High-Recall:** Utilizzo di **YOLOv11x** con risoluzione di input 1088px e soglie di confidenza minime (0.1/0.2) per rilevare piccoli oggetti e giocatori sfocati dal movimento.
* **Field Masking Adattivo:** Algoritmo di segmentazione semantica che combina spazi colore HSV e LAB per identificare dinamicamente il manto erboso e filtrare steward, fotografi e pubblico, riducendo drasticamente i Falsi Positivi.
* **Tracking "Pure Motion":** Implementazione di **BoT-SORT** con *Sparse Optical Flow* (GMC) per la compensazione del movimento di camera. Il sistema non utilizza Re-ID visivo, affidandosi a una logica geometrica robusta (`match_thresh: 0.9`) per evitare scambi di identità tra compagni di squadra.
* **Analisi Comportamentale:** Stima della presenza nelle ROI basata sulla proiezione geometrica del "feet point" (punto di appoggio) dei giocatori.

---

## 📂 Struttura della Repository

```text
├── configs/              # File di configurazione YAML e JSON (tracker, ROI, main)
├── models/               # Pesi dei modelli (YOLO, ReID) e script di conversione
├── output/               # Cartella di output per log, video e risultati
├── SIMULATOR/            # Codice per la simulazione e test locale
├── src/                  # Codice sorgente principale
│   ├── behaviour/        # Modulo per l'analisi comportamentale
│   ├── evaluation/       # Modulo di valutazione (HOTA, nMAE) con TrackEval
│   ├── tracker/          # Logica del tracker (integrazione YOLO + BoT-SORT)
│   ├── utils/            # Utility (Field Masking, BBox operations, visualizzazione)
│   ├── visualizer/       # Generazione video con overlay
│   └── main.py           # Entry point della pipeline
└── tracking/             # Dataset (train, test, challenge)

```

---

## 🛠️ Installazione

1. **Clona la repository:**
```bash
git clone [https://github.com/tuo-username/soccernet-tracking-av2025.git](https://github.com/tuo-username/soccernet-tracking-av2025.git)
cd soccernet-tracking-av2025

```


2. **Installa le dipendenze:**
Si consiglia di utilizzare un ambiente virtuale (venv o conda).
```bash
pip install -r requirements.txt

```


*Dipendenze principali:* `ultralytics`, `opencv-python`, `numpy`, `pyyaml`, `scikit-image`, `pandas`, `trackeval`.

3. **Setup dei Dati:**
Posiziona il dataset SoccerNet nella cartella `tracking/`. La struttura attesa per ogni sequenza è:
```text
tracking/test/SNMOT-XXX/
├── img1/          # Frame JPEG
├── gt/gt.txt      # Ground Truth (opzionale per inferenza)
└── gameinfo.ini   # Metadati

```



---

## 💻 Utilizzo

La pipeline può essere eseguita tramite lo script `src/main.py`. È possibile specificare quali step eseguire (tracking, analisi, visualizzazione, valutazione).

### Esempio Base (Esecuzione Completa)

```bash
python src/main.py --config configs/main_config.yaml --step all --seq all

```

### Argomenti Principali

* `--step`: Scegliere tra `tracking`, `behaviour`, `eval`, `visualizer` o `all`.
* `--seq`: Specificare una o più sequenze (es. `SNMOT-116`) o `all` per elaborare l'intera cartella.
* `--debug`: Attiva la visualizzazione a schermo (`show_tracks`, `show_mask`, `show_behaviour`).

### Configurazione Parametri

I parametri del tracker possono essere modificati nel file `configs/botsort_8.yaml` o passati da riga di comando:

```bash
python src/main.py --conf 0.1 --iou 0.7 --tracker_config configs/botsort.yaml

```

---

## 📊 Risultati e Metriche

Il sistema è stato valutato sul Test Set ufficiale. Di seguito i risultati ottenuti:

| Metrica | Valore | Descrizione |
| --- | --- | --- |
| **HOTA** | **0.742** | Higher Order Tracking Accuracy (Bilanciamento Detection/Association) |
| **DetA** | **0.863** | Detection Accuracy (Precisione del rilevamento) |
| **AssA** | **0.636** | Association Accuracy (Stabilità delle traiettorie) |
| **nMAE** | **0.785** | Normalized Mean Absolute Error (Precisione conteggio ROI) |
| **PTBS** | **1.527** | **Player Tracking and Behavior Score (Score Finale)** |

### Punti di Forza

* **Alta DetA (0.863):** Grazie alla strategia "High Recall", il sistema non perde quasi mai i giocatori, anche in situazioni difficili.
* **AssA Competitiva (0.636):** Nonostante l'assenza di Re-ID, l'uso aggressivo del GMC e soglie di matching strette garantisce un tracciamento stabile.

---

## 📜 Riferimenti

* **YOLO11:** Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11*. [GitHub](https://github.com/ultralytics/ultralytics).
* **SoccerNet:** Deliege, A., et al. (2021). *SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos*.
* **BoT-SORT:** Aharon, N., et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*.

---

*Progetto sviluppato per il corso di Artificial Vision, A.A. 2025/2026.*