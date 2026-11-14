# Customer Support Ticket Router

A bilingual educational intent-classification pipeline using MASSIVE, plus a separate appliance-support test challenge. MASSIVE is a general assistant dataset, not an appliance-support training set. The appliance messages are therefore treated as out of scope. This project is a demonstration, not a production support system.

## Setup

Create and activate a Python environment (Python 3.10 or 3.11 recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the data once, run the NLP tests, and train the models in order:

```bash
python -m src.data.prepare
python -m pytest -q
python -m src.models.baseline
python -m src.models.neural
python -m src.models.teacher
python -m src.models.student
python -m src.eval.evaluate_all
```

The dataset and pretrained transformers require internet access. Transformer training benefits from a GPU; inference and evaluation use CPU. The preparation script recreates the shared split deterministically. Generated data and model artifacts are ignored by Git. Results stay empty until evaluation completes successfully. See `results/results.md` afterward. Model size counts weight files (and the neural word vectors), not tokenizer files or runtime memory. Latency includes tokenization and prediction, and is measured in a warmed-up CPU process. Hardware and metric definitions appear in the results file.

## Run the app

After training all models, use this single command:

```bash
python -m app.main
```

Open the local URL printed by Gradio. The app needs all models for comparison and uses the quantized student for the final route. Predictions below the fixed confidence threshold route to a human; the threshold is illustrative and has not been calibrated. A known intent without a configured team also routes to a human. The rule-based correction covers selected single-word products, not all appliance names or every typo. Model codes are extracted for display and retained in the text sent to classifiers.

See `docs/architecture.md` and `docs/technique_map.md` for how the pieces fit together, and `notebooks/colab_train.ipynb` for a GPU workflow.
