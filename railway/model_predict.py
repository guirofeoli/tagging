# model_predict.py
import json
import joblib
import numpy as np
import os
from git_utils import get_file_from_github
import base64

# Cache (para evitar recarregar a cada request no Railway)
_model = None
_vocab = None
_labels = None

def ensure_file_from_github(filename, binary=False):
    """Se o arquivo não existe localmente, busca do GitHub e salva local."""
    if not os.path.exists(filename):
        content, _ = get_file_from_github(filename)
        if content:
            mode = "wb" if binary else "w"
            with open(filename, mode) as f:
                if binary:
                    f.write(base64.b64decode(content))
                else:
                    f.write(content)

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
    global _model, _vocab, _labels
    # Garante que arquivos existam localmente
    ensure_file_from_github("model.bin", binary=True)
    ensure_file_from_github("allWords.json")
    ensure_file_from_github("labels.json")

    if _model is None:
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)

def predict_session(features):
    """Recebe features (dict) e retorna (label, score/confiança)."""
    _load_all()
    X = np.array([example_to_vector(features, _vocab)])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    label = _labels[idx]
    score = float(probs[idx])
    return label, score

# Teste rápido
if __name__ == "__main__":
    # Exemplo fake para testar
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
