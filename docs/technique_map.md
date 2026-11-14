# Technique map

- Regular expressions normalize repetitions, extract model codes, and match small-talk rules.
- Dynamic programming computes edit distance for product spelling correction.
- TF-IDF and logistic regression form a sparse classical baseline.
- Word2Vec learns word vectors from training text; an SGD-trained feedforward network classifies their average.
- XLM-R is the fine-tuned teacher. The student learns from both the labels and softened teacher predictions.
- Dynamic quantization converts linear layers of the student to INT8 for CPU inference.
- A fixed confidence threshold identifies possible out-of-scope input. Because out-of-scope intents are absent during training, this is rejection rather than a learned out-of-scope class.
- Every model uses the same held-out test data and reports accuracy, out-of-scope recall, weight size and CPU latency.
