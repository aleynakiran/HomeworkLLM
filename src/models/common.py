"""Shared paths and data loading for all training scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts'


def read_split(name):
    with (ROOT / 'data/processed' / f'{name}.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def labels_from_train():
    return sorted({row['label'] for row in read_split('train')})
