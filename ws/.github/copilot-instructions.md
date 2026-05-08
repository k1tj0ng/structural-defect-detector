# Copilot instructions

## Project overview

This repository is a notebook-first computer vision project for concrete defect detection. The working flow is centered on:

- `defect_detection_train.ipynb` for dataset loading, training, fine-tuning, validation, and saving the model artifact.
- `defect_detection_eval.ipynb` for loading `defect_model.keras`, scoring local images, and comparing binary decision thresholds.
- `data/` for image inputs, with the current notebooks operating on `data/Decks`.

## Commands

```bash
pip install -r requirements.txt
jupyter lab
```

Execute the training notebook from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace defect_detection_train.ipynb
```

Execute the evaluation notebook from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace defect_detection_eval.ipynb
```

There are no repository-defined build, lint, or automated test commands, and there is no single-test workflow checked into the repo.

## High-level architecture

`defect_detection_train.ipynb` downloads a Kaggle dataset with `kagglehub`, then trains from a local folder selected via `Path("data/Decks")`. It builds a MobileNetV2 transfer-learning model, trains with the base frozen, fine-tunes by unfreezing the tail of the backbone, computes ROC/AUC on the validation split, and saves the trained model as `defect_model.keras`.

`defect_detection_eval.ipynb` is downstream of that artifact. It loads `defect_model.keras`, reads images from `data/Decks`, predicts defect probabilities for sampled images in `data/Decks/defect` and `data/Decks/no_defect`, and evaluates threshold choices by comparing precision and recall at several cutoffs.

The `data/` directory contains several top-level categories, but the current training and evaluation path is deck-specific. Changes to other categories will not affect the current workflow unless the notebooks are updated to point at them explicitly.

## Key conventions

- Run notebooks from the repository root. Paths are hard-coded as relative paths such as `data/Decks` and `defect_model.keras`.
- Labels are inferred from subdirectory names under `data/Decks`, so preserving the `defect/` and `no_defect/` folder names matters.
- The training notebook includes a binary-or-multiclass output branch, but the evaluation notebook currently assumes binary sigmoid output and reads `model.predict(...)[0][0]`. If the class layout or output head changes, update both notebooks together.
- `defect_model.keras` is the contract between training and evaluation. Renaming or relocating that file requires a matching change in `defect_detection_eval.ipynb`.
- The intended training pattern is transfer learning first, then fine-tuning with a lower learning rate by unfreezing only the last part of the MobileNetV2 backbone.
