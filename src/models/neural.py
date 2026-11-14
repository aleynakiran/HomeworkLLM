"""Train Word2Vec embeddings and an SGD feedforward intent classifier."""
import re
import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn
from src.models.common import ARTIFACTS, labels_from_train, read_split
from src.nlp.normalize import normalize


def tokens(text):
    return re.findall(r'\w+', normalize(text))


def vector(text, words, dimension):
    found = [words[w] for w in tokens(text) if w in words]
    return np.mean(found, axis=0) if found else np.zeros(dimension, dtype=np.float32)


class Network(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, classes))

    def forward(self, x):
        return self.layers(x)


def main():
    torch.manual_seed(42)
    rows = read_split('train')
    labels = labels_from_train()
    embedding = Word2Vec([tokens(r['text']) for r in rows], vector_size=100, min_count=1, workers=1, seed=42)
    x = torch.tensor(np.array([vector(r['text'], embedding.wv, 100) for r in rows]), dtype=torch.float32)
    y = torch.tensor([labels.index(r['label']) for r in rows])
    model = Network(100, len(labels))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(10):
        for indexes in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x[indexes]), y[indexes]).backward()
            optimizer.step()
    ARTIFACTS.mkdir(exist_ok=True)
    embedding.wv.save(str(ARTIFACTS / 'word_vectors.kv'))
    torch.save({'state': model.state_dict(), 'labels': labels, 'dimension': 100}, ARTIFACTS / 'neural.pt')

if __name__ == '__main__':
    main()
