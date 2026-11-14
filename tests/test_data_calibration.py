import csv
from pathlib import Path

from src.data.prepare import APPLIANCE_INTENTS, split_appliances
from src.eval.calibration_utils import choose_threshold


def test_appliance_split_is_disjoint_reproducible_and_stratified():
    path = Path(__file__).resolve().parents[1] / 'data/domain_examples.csv'
    with path.open(encoding='utf-8', newline='') as f:
        examples = list(csv.DictReader(f))
    splits = split_appliances(examples)
    assert splits == split_appliances(examples)
    texts = [{r['text'] for r in splits[name]} for name in ('train', 'val', 'test')]
    assert sum(map(len, texts)) == len(examples)
    assert all(not texts[i] & texts[j] for i in range(3) for j in range(i + 1, 3))
    for rows in splits.values():
        assert {r['label'] for r in rows} == APPLIANCE_INTENTS
        assert all(r['label'] != 'out_of_scope' for r in rows)


def test_calibration_selects_validation_cutoff_and_ties_favor_lower():
    threshold = choose_threshold([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert 0.2 < threshold <= 0.8
    assert choose_threshold([0.5, 0.5], [True, False]) == 0.0


def test_calibration_requires_both_classes():
    try:
        choose_threshold([0.5], [False])
    except ValueError:
        return
    raise AssertionError('Calibration accepted validation data without out-of-scope examples')
