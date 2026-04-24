# Structural Defect Detector — Two-Stage CNN Pipeline

ENGG2112 Project | University of Sydney

A two-stage convolutional neural network pipeline that:
1. **Classifies structure type** — Decks / Pavements / Walls (MobileNetV2, 3-class softmax)
2. **Detects defects** — defect / no_defect per structure type, optimised for **recall** (minimise missed cracks)

A Streamlit demo app lets users upload a photo and get an instant defect verdict with a Grad-CAM attention heatmap.

---

## Results (FAST_MODE — 25% data, 2+2 epochs)

| Structure | Recall | Precision | F1 | ROC-AUC |
|-----------|--------|-----------|-----|---------|
| Decks | 0.767 | 0.341 | 0.472 | 0.846 |
| Pavements | 0.575 | 0.678 | 0.622 | 0.898 |
| Walls | 0.758 | 0.557 | 0.642 | 0.855 |
| **E2E Pipeline** | **0.694** | 0.508 | 0.586 | 0.859 |

> Recall is the primary metric per course requirements — minimise false negatives (missed defects).

---

## Project Structure

```
ws/
├── pipeline_train.ipynb       # Training: data split, Stage 1 + Stage 2 models, threshold tuning
├── pipeline_eval.ipynb        # Evaluation: recall, confusion matrix, ROC-AUC, Grad-CAM
├── app.py                     # Streamlit demo app
├── pipeline_report.md         # Auto-generated metrics summary
├── structure_model.keras      # Trained structure classifier
├── defect_model_decks.keras   # Trained defect model — Decks
├── defect_model_pavements.keras
├── defect_model_walls.keras
├── thresholds.json            # Tuned decision thresholds per structure
├── train_split.csv            # Train/val/test file paths (reproducible splits)
├── val_split.csv
├── test_split.csv
└── requirements.txt
```

---

## Setup

### Option A — Apple Silicon (M1/M2/M3) with GPU acceleration

```bash
# Requires Python 3.11
python3.11 -m venv ws/.venv
source ws/.venv/bin/activate
pip install tensorflow==2.16.2 tensorflow-metal
pip install streamlit pillow scikit-learn pandas matplotlib seaborn
```

### Option B — Standard (CPU)

```bash
pip install -r ws/requirements.txt
```

---

## Dataset

Dataset is not included in this repo (617 MB). Download from Kaggle:

**[Concrete Crack Conglomerate Dataset](https://www.kaggle.com/datasets/)**

Place images in the following layout:

```
ws/data/
├── Decks/
│   ├── defect/       # crack images
│   └── no_defect/
├── Pavements/
│   ├── defect/
│   └── no_defect/
└── Walls/
    ├── defect/
    └── no_defect/
```

---

## Training

Open and run `ws/pipeline_train.ipynb` end-to-end.

- Set `FAST_MODE = True` at the top for a quick test run (25% data, 2+2 epochs, ~5 min on GPU)
- Set `FAST_MODE = False` for full training (~30–60 min on M3 with tensorflow-metal)

Outputs saved to `ws/`: `structure_model.keras`, `defect_model_*.keras`, `thresholds.json`, split CSVs.

---

## Evaluation

Run `ws/pipeline_eval.ipynb` after training. Reports:
- Per-structure recall, confusion matrix, ROC-AUC, F1
- End-to-end pipeline recall vs oracle-structure recall
- Grad-CAM attention heatmaps (model attention, **not** ground-truth crack localisation)

> **Note:** The dataset has image-level labels only — no bounding boxes or segmentation masks. Crack bounding-box localisation is not supported and no synthetic boxes are drawn.

---

## Demo App

```bash
cd ws
source .venv/bin/activate   # or your env
streamlit run app.py
```

Upload a photo of a deck, pavement, or wall. The app classifies structure type, then runs the matching defect model and shows the verdict with a Grad-CAM heatmap.

---

## Evaluation Methodology

Per `lecture_eval_meth.md`:
- **Primary metric:** Recall — minimise false negatives (missed defects are costly)
- **Secondary:** Confusion matrix, ROC-AUC, F1 (guards against precision collapse)
- Threshold tuning: sweep [0.20–0.50], pick lowest threshold with F1 ≥ 0.60
- Class imbalance (10–21% defect): handled via `class_weight` + threshold tuning
