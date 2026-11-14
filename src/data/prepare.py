"""Create reproducible shared splits from MASSIVE and labeled appliance examples."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED = 42
APPLIANCE_INTENTS = {'appliance_fault', 'appliance_usage', 'spare_parts', 'service_appointment', 'warranty_returns'}


def split_appliances(examples):
    """Stratify by intent; reserve at least one example for validation and test."""
    groups = defaultdict(list)
    for row in examples:
        if row['intent'] not in APPLIANCE_INTENTS or row['locale'] not in ('tr-TR', 'en-US') or not row['text'].strip():
            raise ValueError(f'Invalid appliance example: {row}')
        groups[row['intent']].append(row)
    if set(groups) != APPLIANCE_INTENTS:
        raise ValueError('Every appliance intent must have examples')
    rng = random.Random(SEED)
    splits = {'train': [], 'val': [], 'test': []}
    for intent in sorted(groups):
        group = groups[intent][:]
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f'Need at least three examples for {intent}')
        val_count = max(1, round(len(group) * 0.15))
        test_count = max(1, round(len(group) * 0.15))
        for split, part in [('val', group[:val_count]), ('test', group[val_count:val_count + test_count]), ('train', group[val_count + test_count:])]:
            splits[split].extend({'text': r['text'], 'label': intent, 'locale': r['locale']} for r in part)
    return splits


def main():
    # Imported here so the split helper can be tested without downloading MASSIVE.
    from datasets import load_dataset
    import yaml
    config = yaml.safe_load((ROOT / 'configs/teams.yaml').read_text())
    held_out = set(config['held_out_intents'])
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'data/domain_examples.csv').open(newline='', encoding='utf-8') as f:
        rows = split_appliances(list(csv.DictReader(f)))
    observed = set()
    for locale in ('tr-TR', 'en-US'):
        dataset = load_dataset('AmazonScience/massive', locale, trust_remote_code=True)
        names = dataset['train'].features['intent'].names
        observed.update(names)
        for source, target in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            for example in dataset[source]:
                intent = names[example['intent']]
                if intent in held_out and target == 'train':
                    continue
                rows[target].append({'text': example['utt'], 'label': 'out_of_scope' if intent in held_out else intent, 'locale': locale})
    missing = held_out - observed
    if missing:
        raise ValueError(f'Unknown held-out intents: {sorted(missing)}')
    for split, examples in rows.items():
        random.Random(SEED).shuffle(examples)
        with (output / f'{split}.jsonl').open('w', encoding='utf-8') as f:
            for row in examples:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('Prepared shared files in', output)

if __name__ == '__main__':
    main()
