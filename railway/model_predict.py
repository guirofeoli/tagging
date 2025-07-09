# model_predict.py - atualizado para features enriquecidas
import json
import joblib
import numpy as np
import os

# Campos numéricos e de cor/estilo (devem bater com o train_ux.py)
NUMERIC_FIELDS = ["width", "height", "fontSize", "y", "siblingIndex"]
STYLE_FIELDS = ["bgColor", "fontWeight", "fontColor"]

# Cache (para evitar recarregar a cada request)
_model = None
_vocab = None
_labels = None
_numeric_means = None
_numeric_stds = None

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
    # Campos extra de estilo/cor
    for k in STYLE_FIELDS:
        if features.get(k):
            tokens.append(f"{k}={features[k]}")
    return tokens

def example_to_vector(example, vocab, numeric_means=None, numeric_stds=None):
    tokens = set(extract_tokens(example))
    vector = [1 if w in tokens else 0 for w in vocab]
    for f in NUMERIC_FIELDS:
        val = float(example.get(f, 0) or 0)
        if numeric_means and numeric_stds and f in numeric_means:
            std = numeric_stds[f] if numeric_stds[f] > 0 else 1
            val = (val - numeric_means[f]) / std
        vector.append(val)
    return vector

def _load_all():
    global _model, _vocab, _labels, _numeric_means, _numeric_stds
    if _model is None:
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)
    if _numeric_means is None:
        with open("numeric_means.json", "r", encoding="utf-8") as f:
            _numeric_means = json.load(f)
    if _numeric_stds is None:
        with open("numeric_stds.json", "r", encoding="utf-8") as f:
            _numeric_stds = json.load(f)

def predict_session(features):
    """Recebe features (dict) e retorna (label, score/confiança)."""
    _load_all()
    X = np.array([example_to_vector(features, _vocab, _numeric_means, _numeric_stds)])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    label = _labels[idx]
    score = float(probs[idx])
    return label, score

# Teste rápido
if __name__ == "__main__":
    example = {
        "class": "menu superior",
        "text": "Início",
        "id": "menu-inicio",
        "tag": "A",
        "parents": [{"tag": "NAV", "class": "navbar", "id": "nav-main", "text": ""}],
        "contextHeadings": ["Menu"],
        "y": 8,
        "siblingIndex": 0,
        "width": 130,
        "height": 34,
        "bgColor": "#ffffff",
        "fontWeight": 400,
        "fontSize": 15,
        "fontColor": "#333333"
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
