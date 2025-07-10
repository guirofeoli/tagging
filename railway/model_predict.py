# model_predict.py
import json
import joblib
import numpy as np
import os

_model = None
_vocab = None
_labels = None
_numeric_means = None

# Features numéricas que o modelo pode esperar
NUMERIC_KEYS = [
    "length", "y", "siblingIndex", "width", "height", "visible",
    "depth", "numChildren"
]

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
    # Features categóricas importantes (cor, font, etc) podem ser concatenadas
    if 'bgColor' in features and features['bgColor']:
        tokens.append("bgc_" + features['bgColor'])
    if 'textColor' in features and features['textColor']:
        tokens.append("txtc_" + features['textColor'])
    if 'fontWeight' in features and features['fontWeight']:
        tokens.append("fw_" + str(features['fontWeight']))
    if 'fontSize' in features and features['fontSize']:
        tokens.append("fs_" + str(features['fontSize']))
    return tokens

def example_to_vector(example, vocab, numeric_means=None):
    # One-hot tokens
    tokens = set(extract_tokens(example))
    vec = [1 if w in tokens else 0 for w in vocab]

    # Features numéricas padronizadas
    if numeric_means is not None:
        for key in NUMERIC_KEYS:
            val = example.get(key, None)
            try:
                val = float(val)
            except:
                val = None
            mean = numeric_means.get(key, 0.0)
            if val is None:
                val = mean
            # Subtrai a média (normalização centrada)
            vec.append(val - mean)
    return vec

def _load_all():
    global _model, _vocab, _labels, _numeric_means
    if _model is None:
        if not os.path.exists("model.bin"):
            raise FileNotFoundError("Arquivo 'model.bin' não encontrado (treine e exporte o modelo primeiro)!")
        _model = joblib.load("model.bin")
    if _vocab is None:
        if not os.path.exists("allWords.json"):
            raise FileNotFoundError("Arquivo 'allWords.json' não encontrado!")
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        if not os.path.exists("labels.json"):
            raise FileNotFoundError("Arquivo 'labels.json' não encontrado!")
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)
    if _numeric_means is None:
        if not os.path.exists("numeric_means.json"):
            raise FileNotFoundError("Arquivo 'numeric_means.json' não encontrado!")
        with open("numeric_means.json", "r", encoding="utf-8") as f:
            _numeric_means = json.load(f)

def predict_session(features):
    """Recebe features (dict) e retorna (label, score/confiança)."""
    _load_all()
    X = np.array([example_to_vector(features, _vocab, _numeric_means)])
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
        "width": 100,
        "height": 50,
        "bgColor": "rgb(255,255,255)",
        "textColor": "rgb(32,37,42)",
        "fontWeight": "400",
        "fontSize": "16px",
        "visible": 1,
        "depth": 5,
        "numChildren": 0
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
