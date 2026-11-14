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
    G --> L[Comparison table]
    H --> L
    I --> L
    J --> L
    K --> L
    K --> M{Confidence and team mapping}
    M -->|low confidence or unknown team| N[Human agent]
    M -->|mapped intent| O[Support team]
```

Preparation excludes held-out intents from train and validation, relabels them in test, and adds appliance messages to test. All evaluation models use that test file and the same confidence rule. Preprocessing in the app precedes model inference; the model code is displayed without being removed from the message.
