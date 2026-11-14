# Presentation script

Hello! This project routes Turkish and English support messages.

The data pipeline prepares shared training, validation, and test files. Some intents are held out to explore how the system handles unfamiliar requests. Appliance examples add a separate domain challenge.

The app normalizes the message, extracts a model code, and corrects selected product spelling mistakes. Simple greetings and thanks receive a rule-based response.

For other messages, the interface compares a classical baseline, a neural network, a transformer teacher, and a distilled student with its quantized version. The quantized student determines the final route. Low-confidence requests go to a human agent.

The evaluation script uses the same test set for each model and records performance and hardware measurements. The results should be discussed only after the script has run.
