"""Train a normalized TF-IDF and logistic regression baseline."""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.models.common import ARTIFACTS, read_split
from src.nlp.normalize import normalize


def main():
    rows = read_split('train')
    model = make_pipeline(TfidfVectorizer(preprocessor=normalize, lowercase=False, ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    model.fit([r['text'] for r in rows], [r['label'] for r in rows])
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / 'baseline.pkl').open('wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
