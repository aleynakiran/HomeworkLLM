"""Measure every model against the same test file on CPU."""
import json
import os
import platform
import time
from pathlib import Path
from src.models.common import ARTIFACTS, ROOT, read_split
from src.models.inference import NAMES, Predictor
from src.nlp.edit_distance import correct_products
from src.nlp.normalize import normalize


def artifact_size(name):
    if name == 'baseline':
        paths = [ARTIFACTS / 'baseline.pkl']
    elif name == 'neural':
        paths = [ARTIFACTS / 'neural.pt', *ARTIFACTS.glob('word_vectors.kv*')]
    elif name == 'student_int8':
        paths = [ARTIFACTS / 'student_int8.pt']
    else:
        path = ARTIFACTS / name
        paths = [p for p in path.rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.bin')]
    if not paths:
        raise FileNotFoundError(f'No model weights for {name}')
    return sum(os.path.getsize(p) for p in paths) / (1024 * 1024)


def main():
    rows = read_split('test')
    if not rows:
        raise ValueError('The shared test set is empty')
    hardware = {'platform': platform.platform(), 'processor': platform.processor(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count()}
    results = []
    for name in NAMES:
        model = Predictor(name)
        texts = [correct_products(normalize(r['text'])) for r in rows]
        predictions = [model.predict(text)[0] for text in texts]
        correct = sum(p == row['label'] for p, row in zip(predictions, rows))
        oos = [p for p, row in zip(predictions, rows) if row['label'] == 'out_of_scope']
        if not oos:
            raise ValueError('Test set has no out-of-scope examples')
        model.predict(texts[0])  # Warm up before timing.
        start = time.perf_counter()
        for i in range(100):
            model.predict(texts[i % len(texts)])
        latency = (time.perf_counter() - start) * 1000 / 100
        results.append({'model': name, 'accuracy': correct / len(rows), 'out_of_scope_recall': sum(p == 'out_of_scope' for p in oos) / len(oos), 'model_size_mb': artifact_size(name), 'latency_ms': latency})
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    (output / 'results.json').write_text(json.dumps({'hardware': hardware, 'test_set': 'data/processed/test.jsonl', 'results': results}, indent=2) + '\n')
    lines = ['# Evaluation', '', f'CPU hardware: {hardware}', '', '| Model | Accuracy | Out-of-scope recall | Model size (MB) | Mean CPU latency (ms) |', '| --- | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['out_of_scope_recall']:.4f} | {r['model_size_mb']:.2f} | {r['latency_ms']:.2f} |")
    (output / 'results.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
