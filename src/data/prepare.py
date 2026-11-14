"""Download MASSIVE and create the one shared train/validation/test split."""
import csv
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]


def main():
    config = yaml.safe_load((ROOT / 'configs/teams.yaml').read_text())
    held_out = set(config['held_out_intents'])
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    rows = {'train': [], 'val': [], 'test': []}
    observed = set()
    for locale in ('tr-TR', 'en-US'):
        dataset = load_dataset('AmazonScience/massive', locale, trust_remote_code=True)
        names = dataset['train'].features['intent'].names
        observed.update(names)
        for source, target in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            for example in dataset[source]:
                intent = names[example['intent']]
                if intent in held_out and target != 'test':
                    continue
                rows[target].append({'text': example['utt'], 'label': 'out_of_scope' if intent in held_out else intent, 'locale': locale})
    missing = held_out - observed
    if missing:
        raise ValueError(f'Unknown held-out intents: {sorted(missing)}')
    # Handwritten examples are a separate out-of-domain test challenge, never training data.
    with (ROOT / 'data/domain_examples.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows['test'].append({'text': row['text'], 'label': 'out_of_scope', 'locale': row['locale']})
    for split, examples in rows.items():
        random.Random(42).shuffle(examples)
        with (output / f'{split}.jsonl').open('w', encoding='utf-8') as f:
            for row in examples:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('Prepared shared files in', output)

if __name__ == '__main__':
    main()
