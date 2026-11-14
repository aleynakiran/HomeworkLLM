# Customer Support Ticket Router plan

Build each stage in order and commit it separately. Keep generated datasets, trained models and measured results reproducible.

1. Data: configure teams and held-out intents, write appliance examples, and prepare a single shared split.
2. NLP: implement normalization, code extraction, spelling correction and small talk; add tests.
3. Baseline: train TF-IDF and logistic regression.
4. Neural network: train word embeddings and a feedforward classifier.
5. Teacher: fine-tune multilingual XLM-R.
6. Student: distill the teacher and dynamically quantize the student.
7. Evaluation: measure every model on the shared test set, including CPU latency and file sizes.
8. App: show all predictions and route using the quantized student.
9. Documentation: explain architecture, techniques, AI usage and presentation.
10. Colab: run the full pipeline with GPU support.

Do not invent metrics. Keep model artifacts out of Git and record results only when evaluation actually runs.
