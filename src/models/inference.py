"""Load models once and provide a common CPU prediction interface."""
import json
import pickle
import numpy as np
import torch
from gensim.models import KeyedVectors
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from src.models.common import ARTIFACTS
from src.models.neural import Network, vector
from src.models.quantization import select_quantized_engine
from src.nlp.normalize import normalize

NAMES = ('baseline', 'neural', 'teacher', 'student', 'student_int8')


def load_thresholds():
    path = ARTIFACTS / 'confidence_thresholds.json'
    if not path.exists():
        raise FileNotFoundError('Run python -m src.eval.calibrate after training all models')
    thresholds = json.loads(path.read_text(encoding='utf-8'))['thresholds']
    if set(thresholds) != set(NAMES) or any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError('Invalid confidence thresholds; rerun calibration')
    return thresholds


class Predictor:
    def __init__(self, name):
        self.name = name
        if name == 'baseline':
            with (ARTIFACTS / 'baseline.pkl').open('rb') as f:
                self.model = pickle.load(f)
        elif name == 'neural':
            checkpoint = torch.load(ARTIFACTS / 'neural.pt', map_location='cpu', weights_only=True)
            self.labels, self.dimension = checkpoint['labels'], checkpoint['dimension']
            self.model = Network(self.dimension, len(self.labels))
            self.model.load_state_dict(checkpoint['state'])
            self.model.eval()
            self.words = KeyedVectors.load(str(ARTIFACTS / 'word_vectors.kv'), mmap='r')
        else:
            path = ARTIFACTS / ('teacher' if name == 'teacher' else 'student')
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            if name == 'student_int8':
                select_quantized_engine(torch.backends.quantized)
                self.model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(path))
                self.model = torch.quantization.quantize_dynamic(self.model.eval(), {torch.nn.Linear}, dtype=torch.qint8)
                self.model.load_state_dict(torch.load(ARTIFACTS / 'student_int8.pt', map_location='cpu', weights_only=True))
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.eval()
            self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_raw(self, text):
        text = normalize(text)
        with torch.inference_mode():
            if self.name == 'baseline':
                probabilities = self.model.predict_proba([text])[0]
                labels = self.model.classes_
            elif self.name == 'neural':
                x = torch.tensor(np.array([vector(text, self.words, self.dimension)]), dtype=torch.float32)
                probabilities = self.model(x).softmax(-1)[0].numpy()
                labels = self.labels
            else:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
                probabilities = self.model(**inputs).logits.softmax(-1)[0].numpy()
                labels = self.labels
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        return str(labels[index]), confidence

    def predict(self, text, threshold):
        label, confidence = self.predict_raw(text)
        return (label if confidence >= threshold else 'out_of_scope', confidence)
