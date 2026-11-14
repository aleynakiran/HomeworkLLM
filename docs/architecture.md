# Architecture

```mermaid
flowchart TD
    A[Message] --> B[Turkish-aware normalization]
    B --> C[Extract model code]
    B --> D[Correct selected product words]
    D --> E{ELIZA small talk match?}
    E -->|yes| F[Return reply, model code and Small talk route]
    E -->|no| G[Baseline prediction and threshold]
    E -->|no| H[Embedding network prediction and threshold]
    E -->|no| I[Teacher prediction and threshold]
    E -->|no| J[Student prediction and threshold]
    E -->|no| K[Quantized student prediction and threshold]
    T[Load calibrated thresholds at startup] --> G
    T --> H
    T --> I
    T --> J
    T --> K
    G --> L[Comparison table: intent, confidence, latency]
    H --> L
    I --> L
    J --> L
    K --> L
    K --> M{Out of scope or unmapped?}
    M -->|yes| N[Human agent]
    M -->|no| O[Mapped support team]
    C --> P[Display model code]
    D --> Q[Display corrected message when not small talk]
```

The app loads all models and their validation-calibrated thresholds at startup. Each model normalizes its input again before predicting; the app applies product correction before inference. The model code is displayed without being removed from the input. The quantized student's thresholded prediction determines the route; mappings use an exact intent or its prefix.

The data preparation script combines a fixed-seed, intent-stratified appliance split with the original MASSIVE splits. It excludes held-out MASSIVE intents from training and labels those intents `out_of_scope` in validation and test. Every model learns from the combined training split. After training, calibration selects one threshold per model using validation confidence scores. Evaluation uses these thresholds and the same test file for every model, then saves up to ten quantized-student errors in test order.
