# Technique map

- A fixed-seed, intent-stratified split puts labeled appliance examples into training, validation, and test alongside the original MASSIVE splits. Held-out MASSIVE intents are excluded from training and labeled out of scope in validation and test.
- Regular expressions normalize repetitions, extract model codes, and match small-talk rules.
- Dynamic programming computes edit distance for product spelling correction.
- TF-IDF and logistic regression form a sparse classical baseline.
- Word2Vec learns word vectors from training text; an SGD-trained feedforward network classifies their average.
- XLM-R is the fine-tuned teacher. The student learns from both the labels and softened teacher predictions.
- Dynamic quantization converts linear layers of the student to INT8 for CPU inference.
- Each model's threshold maximizes the mean of in-scope acceptance and held-out-intent rejection on validation. Ties favor the lower threshold. A prediction below the threshold is rejected as out of scope; this is not a learned out-of-scope class.
- Every model is evaluated on the same test data for accuracy, out-of-scope recall, weight size, and CPU latency. The quantized student's calibrated prediction determines the app's route.
