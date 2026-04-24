# Claude Instruction

First, inspect the dataset in `ws/data` and the evaluation rules in `lecture_eval_meth.md`.

Then plan and execute a CNN-based pipeline with the following goals:

1. Classify the data type first:
   - `Decks`
   - `Pavements`
   - `Walls`

2. Classify the defect status next:
   - `defect`
   - `no_defect`

3. Optimize for recall as the primary metric.
   - Minimize false negatives.
   - Report recall first.
   - Also report confusion matrix, ROC-AUC, and F1 score as secondary checks.

4. If bounding-box or segmentation annotations exist, add crack localization.
   - Draw a red box around the crack area.
   - If no localization labels exist, do not invent boxes.
   - In that case, state clearly that crack boxing is not supported by the available labels.

5. Produce the full workflow.
   - Data loading and splitting
   - Model design
   - Training
   - Evaluation
   - Threshold tuning if needed
   - Clear assumptions and limitations

Use the dataset structure and evaluation method as the source of truth. Do not optimize for accuracy alone.
