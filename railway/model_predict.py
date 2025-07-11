# model_predict.py — Pipeline híbrida: heurísticas + ML + fallback

import json
import joblib
import numpy as np
import os

_model = None
_vocab = None
_labels = None

def extract_tokens(features):
    tokens = []
    for k in ['class', 'text', 'id', 'tag', 'aria', 'role', 'href', 'type']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'parents' in features:
        for p in features['parents']:
            for k in ['class', 'text', 'id', 'tag', 'aria', 'role', 'href', 'type']:
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
    if _model is None:
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)

# --------- HEURÍSTICAS UNIVERSAIS (pode expandir depois) ---------
def regras_classicas(features):
    tag = (features.get("tag") or "").lower()
    classe = (features.get("class") or "").lower()
    texto = (features.get("text") or "").lower()
    id_ = (features.get("id") or "").lower()

    # 1. Menu/Navigation
    if tag == "nav" or "menu" in classe or "nav" in classe or "menu" in id_:
        return ("Menu", 0.99)
    # 2. Header
    if tag == "header" or "header" in classe or "topo" in texto or "cabeçalho" in texto:
        return ("Header", 0.99)
    # 3. Footer
    if tag == "footer" or "footer" in classe or "rodape" in texto or "rodapé" in texto:
        return ("Rodape", 0.99)
    # 4. Hero
    if "hero" in classe or "hero" in id_:
        return ("Hero", 0.97)
    # 5. Modal/Dialog
    if "modal" in classe or "modal" in id_ or tag == "dialog":
        return ("Modal", 0.98)
    # 6. Botão Fechar
    if "fechar" in texto or "close" in texto:
        return ("Fechar", 0.96)
    # 7. Conteúdo/Content
    if tag in ["main", "section", "article"]:
        return ("Conteudo", 0.95)
    # 8. Se tiver role navigation/footer/etc
    role = (features.get("role") or "").lower()
    if role == "navigation":
        return ("Menu", 0.97)
    if role == "contentinfo":
        return ("Rodape", 0.97)
    # 9. Se o texto é "comprar" ou "buy"
    if "comprar" in texto or "buy" in texto:
        return ("Comprar", 0.96)
    # Expanda aqui com outros padrões

    return None

# --------- PIPELINE PREDITOR ---------
def predict_session(features):
    # 1. Heurísticas
    heuristica = regras_classicas(features)
    if heuristica:
        label, score = heuristica
        return label, score

    # 2. Modelo ML (se disponível)
    try:
        _load_all()
        X = np.array([example_to_vector(features, _vocab)])
        probs = _model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        label = _labels[idx]
        score = float(probs[idx])
        return label, score
    except Exception as e:
        # Fallback
        return "[Auto-UX] Não rotulado", 0.0

# Teste rápido
if __name__ == "__main__":
    example = {
        "tag": "NAV", "class": "navbar menu-principal", "id": "nav-main", "text": "",
        "parents": [{"tag": "BODY", "class": "", "id": "", "text": ""}],
        "contextHeadings": ["Menu"],
        "y": 5, "siblingIndex": 0
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
