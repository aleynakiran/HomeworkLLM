# Customer Support Ticket Router

A bilingual educational intent-classification pipeline combining MASSIVE with labeled appliance-support messages. Appliance requests are in scope, with intents for faults, usage, spare parts, service appointments, and warranty or returns. This is a demonstration, not a production support system.

## Setup

Create and activate a Python environment (Python 3.10 or 3.11 recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the data, run the tests, train all models, calibrate thresholds, and evaluate:

```bash
python -m src.data.prepare
python -m pytest -q
python -m src.models.baseline
python -m src.models.neural
python -m src.models.teacher
python -m src.models.student
python -m src.eval.calibrate
python -m src.eval.evaluate_all
```

Rerun the entire sequence after changing the examples or labels: existing processed data and trained models do not update themselves. The dataset and pretrained transformers require internet access. Transformer training benefits from a GPU; inference, calibration, and evaluation use CPU.

The preparation script uses a fixed seed to split appliance examples by intent into train, validation, and test. MASSIVE uses its original splits. The held-out MASSIVE intents are excluded from training and labeled `out_of_scope` in validation and test; appliance examples are never assigned that label. Each model trains on the combined training set. The validation split contains both in-scope and held-out examples; it is not used for model training.

Calibration chooses a separate confidence threshold for each model from validation predictions. It maximizes the mean of in-scope acceptance and out-of-scope rejection, breaking ties in favor of the lower threshold. This objective measures rejection, not intent accuracy, and the threshold is not a guarantee on new requests. Thresholds are saved in `artifacts/confidence_thresholds.json`. Recalibrate after retraining; the app and evaluation require that file and never silently fall back to a default.

Generated data and model artifacts are ignored by Git. Evaluation uses the same `data/processed/test.jsonl` for all models and writes measured results to `results/results.json` and `results/results.md`. Model size counts weight files (and the neural word vectors), not tokenizer files or runtime memory. Latency includes tokenization and prediction, measured in a warmed-up CPU process; hardware is recorded with the results.

## Run the app

After completing the sequence above, use this single command:

```bash
python -m app.main
```

Open the local URL printed by Gradio. The app loads all models for comparison and uses the quantized student's prediction and calibrated threshold for the final route. Low-confidence predictions and unmapped intents route to a human. Product correction covers selected single-word products. Model codes are displayed and retained in the text sent to classifiers.

See `docs/architecture.md`, `docs/technique_map.md`, and `notebooks/colab_train.ipynb` for the GPU workflow.
