"""Choose a separate threshold for each model using only the validation split."""
import json
from src.eval.calibration_utils import choose_threshold
from src.models.common import ARTIFACTS, read_split
from src.models.inference import NAMES, Predictor
from src.nlp.edit_distance import correct_products
from src.nlp.normalize import normalize


def main():
    rows = read_split('val')
    texts = [correct_products(normalize(row['text'])) for row in rows]
    out_of_scope = [row['label'] == 'out_of_scope' for row in rows]
    thresholds = {}
    for name in NAMES:
        model = Predictor(name)
        confidences = [model.predict_raw(text)[1] for text in texts]
        thresholds[name] = choose_threshold(confidences, out_of_scope)
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / 'confidence_thresholds.json'
    path.write_text(json.dumps({'validation_set': 'data/processed/val.jsonl', 'objective': 'mean in-scope acceptance and out-of-scope rejection', 'thresholds': thresholds}, indent=2) + '\n')
    print('Saved calibration to', path)

if __name__ == '__main__':
    main()
