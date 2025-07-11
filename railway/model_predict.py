import json
import joblib
import numpy as np

_model = None
_vocab = None
_labels = None
_tfidf = None
_examples_texts = None

def flatten_features(example):
    text_parts = [
        example.get('text',''),
        example.get('tag',''),
        example.get('id',''),
        example.get('class',''),
        example.get('cssPath',''),
        str(example.get('contextHeadings',[])),
    ]
    for p in example.get('parents',[]):
        text_parts.append(p.get('text',''))
        text_parts.append(p.get('cssPath',''))
    return " | ".join([str(x) for x in text_parts if x])

def extract_tokens(features):
    tokens = []
    for k in ['class', 'text', 'id', 'tag', 'cssPath', 'role', 'type', 'placeholder', 'href', 'title', 'alt']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'parents' in features:
        for p in features['parents']:
            for k in ['class', 'text', 'id', 'tag', 'cssPath']:
                val = p.get(k, "")
                if isinstance(val, str):
                    tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'contextHeadings' in features and features['contextHeadings']:
        for h in features['contextHeadings']:
            tokens += [w.lower() for w in h.split() if len(w) > 1]
    if 'y' in features: tokens.append(f'y{int(features["y"]//10)*10}')
    if 'siblingIndex' in features: tokens.append(f'sib{min(int(features["siblingIndex"]), 10)}')
    if 'width' in features: tokens.append(f'w{int(features["width"]//50)*50}')
    if 'height' in features: tokens.append(f'h{int(features["height"]//20)*20}')
    if 'depth' in features: tokens.append(f'd{min(int(features["depth"]), 10)}')
    if 'clickable' in features: tokens.append(f'click{features["clickable"]}')
    return tokens

def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

def _load_all():
    global _model, _vocab, _labels, _tfidf, _examples_texts
    if _model is None:
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)
    if _tfidf is None:
        _tfidf = joblib.load("model_tfidf.bin")
    if _examples_texts is None:
        with open("all_examples_texts.json", "r", encoding="utf-8") as f:
            _examples_texts = json.load(f)

def heuristics(features):
    tag = (features.get('tag','') or '').lower()
    y = features.get('y',0)
    role = (features.get('role','') or '').lower()
    headings = [h.lower() for h in features.get('contextHeadings',[])]
    # Heurísticas rápidas!
    if tag in ("nav",) or "menu" in features.get("id","").lower() or "menu" in features.get("class","").lower():
        return "Menu"
    if tag in ("footer",) or y > 80: return "Footer"
    if tag in ("header",) or y < 18: return "Header"
    if tag in ("main","section","article"): return "Conteúdo"
    if role and "banner" in role: return "Banner"
    if tag in ("h1","h2","h3","h4","h5","h6") and features.get("text",""): return "Heading"
    for h in headings:
        if "hero" in h: return "Hero"
        if "menu" in h: return "Menu"
        if "footer" in h: return "Footer"
        if "header" in h: return "Header"
    return None

def tfidf_predict(features, labels):
    # Similaridade textual entre features do elemento e exemplos rotulados
    global _tfidf, _examples_texts
    query = flatten_features(features)
    v_query = _tfidf.transform([query])
    v_corpus = _tfidf.transform(_examples_texts)
    sims = np.dot(v_corpus, v_query.T).toarray().flatten()
    idx = np.argmax(sims)
    return labels[idx], float(sims[idx])

def predict_session(features):
    _load_all()
    # 1. Heurísticas simples
    h = heuristics(features)
    if h: return h, 1.0

    # 2. Random Forest
    X = np.array([example_to_vector(features, _vocab)])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    label = _labels[idx]
    score = float(probs[idx])
    if score > 0.7:
        return label, score

    # 3. Similaridade textual (TF-IDF)
    label2, score2 = tfidf_predict(features, _labels)
    if score2 > 0.6:
        return label2, score2

    # 4. Fallback
    return "Outro", 0.0

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
        "cssPath": "nav#nav-main > ul.menu > li",
        "clickable": 1
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
