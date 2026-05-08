# XGBoost Pipeline Report

*Generated: 2026-05-08 23:06*

## Assumptions & Limitations

- Features are 1280-dim MobileNetV2 GlobalAveragePooling2D embeddings (frozen backbone, no fine-tuning).
- XGBoost has no spatial attention; SHAP values show embedding-dimension importance, not pixel-level localisation.
- Splits are identical to the CNN pipeline (`train_split.csv`, `val_split.csv`, `test_split.csv`).

## Data Split

| Split | Images |
|---|---|
| Train | 38,809 |
| Val   | 8,543 |
| Test  | 8,740 |

## Stage 1 — Structure Classifier

Test accuracy: **0.908**

## Stage 2 — Per-Structure Defect Classifiers (Oracle Routing)

| Structure   |   Threshold |   Recall |   Precision |    F1 |   ROC-AUC |   TN |   FP |   FN |   TP |
|:------------|------------:|---------:|------------:|------:|----------:|-----:|-----:|-----:|-----:|
| Decks       |        0.25 |    0.461 |       0.432 | 0.446 |     0.756 | 1759 |  192 |  171 |  146 |
| Pavements   |        0.3  |    0.428 |       0.248 | 0.314 |     0.721 | 2824 |  495 |  218 |  163 |
| Walls       |        0.25 |    0.529 |       0.373 | 0.437 |     0.684 | 1561 |  570 |  302 |  339 |

## End-to-End Pipeline

| Metric | Value |
|---|---|
| Recall | 0.457 |
| Precision | 0.328 |
| F1 | 0.382 |
| ROC-AUC | 0.719 |
| Oracle Recall | 0.484 |
| Recall Drop | 0.027 |
| False Negatives | 727 |

## Tuned Thresholds

- Decks: 0.25
- Pavements: 0.3
- Walls: 0.25

---
*XGBoost + MobileNetV2 feature extraction pipeline*
