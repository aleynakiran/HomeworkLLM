# Architecture

```mermaid
flowchart TD
    A[Message] --> B[Turkish-aware normalization]
    B --> C[Extract model code for display]
    B --> D[Correct selected product words]
    D --> E{ELIZA small talk?}
    E -->|yes| F[Return rule-based reply]
    E -->|no| G[Baseline prediction]
    E -->|no| H[Word embedding network prediction]
    E -->|no| I[Teacher prediction]
    E -->|no| J[Student prediction]
    E -->|no| K[Quantized student prediction]
    T[Validation-calibrated thresholds] --> G
    T --> H
    T --> I
    T --> J
    T --> K
    G --> L[Comparison table]
    H --> L
    I --> L
    J --> L
    K --> L
    K --> M{Team mapping}
    M -->|out of scope or unmapped| N[Human agent]
    M -->|mapped intent| O[Support team]
```

The data preparation script combines a fixed-seed, intent-stratified appliance split with the original MASSIVE splits. It excludes held-out MASSIVE intents from training and labels those intents `out_of_scope` in validation and test. Every model learns from the combined training split. After training, calibration selects one threshold per model using validation confidence scores. Evaluation uses these thresholds and the same test file for every model. In the app, the quantized student determines the route. The extracted code is displayed without being removed from the message.
