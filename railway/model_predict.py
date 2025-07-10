import json
import joblib
import numpy as np
import os

_model = None
_vocab = None
_labels = None
_numeric = None

def extract_tokens(features):
    tokens = []
    for k in ['class', 'text', 'id', 'tag']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'parents' in features:
        for p in features['parents']:
            for k in ['class', 'text', 'id', 'tag']:
                val = p.get(k, "")
                if isinstance(val, str):
                    tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'contextHeadings' in features and features['contextHeadings']:
        for h in features['contextHeadings']:
            tokens += [w.lower() for w in h.split() if len(w) > 1]
    if 'y' in features:
        tokens.append(f'y{int(features["y"]//10)*10}')
    if 'siblingIndex' in features:
        tokens.append(f'sib{min(int(features["siblingIndex"]), 10)}')
    return tokens

def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

def _load_all():
    global _model, _vocab, _labels, _numeric
    if _model is None:
        if not os.path.exists("model.bin"):
            raise FileNotFoundError("Arquivo 'model.bin' não encontrado (treine e exporte o modelo primeiro)!")
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)
    if _numeric is None and os.path.exists("numeric_means.json"):
        with open("numeric_means.json", "r", encoding="utf-8") as f:
            _numeric = json.load(f)

def predict_session(features):
    _load_all()
    X = np.array([example_to_vector(features, _vocab)])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    label = _labels[idx]
    score = float(probs[idx])
    return label, score

if __name__ == "__main__":
    example = {
        "class": "menu superior",
        "text": "Início",
        "id": "menu-inicio",
        "tag": "A",
        "parents": [{"tag": "NAV", "class": "navbar", "id": "nav-main", "text": ""}],
        "contextHeadings": ["Menu"],
        "y": 8,
        "siblingIndex": 0
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
