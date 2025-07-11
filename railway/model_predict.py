import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cache global (para Railway)
_rf = None
_clf_emb = None
_label_encoder = None
_vocab = None
_embs = None
_examples_y = None
_embedder = None

def load_all():
    global _rf, _clf_emb, _label_encoder, _vocab, _embs, _examples_y, _embedder
    if _rf is None:
        _rf = joblib.load("rf_model.bin")
    if _clf_emb is None:
        _clf_emb = joblib.load("emb_model.bin")
    if _label_encoder is None:
        _label_encoder = joblib.load("label_encoder.bin")
    if _vocab is None:
        with open("allWords.json", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _embs is None:
        _embs = np.load("embeddings.npy")
    if _examples_y is None:
        with open("examples_meta.json", encoding="utf-8") as f:
            _examples_y = json.load(f)
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        tokens.append(f'y{int(features['y']//10)*10}')
    if 'siblingIndex' in features:
        tokens.append(f'sib{min(int(features['siblingIndex']), 10)}')
    return tokens

def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

def features_to_text(features):
    txt = [
        features.get('text', ''),
        features.get('class', ''),
        features.get('id', ''),
        features.get('tag', '')
    ] + features.get('contextHeadings', [])
    return " ".join([t for t in txt if t])

def predict_session(features):
    load_all()
    # 1. Features tradicionais (Random Forest)
    X_rf = np.array([example_to_vector(features, _vocab)])
    rf_pred = _rf.predict(X_rf)[0]
    rf_prob = max(_rf.predict_proba(X_rf)[0])
    rf_label = _label_encoder.inverse_transform([rf_pred])[0]

    # 2. Embedding classifier
    txt = features_to_text(features)
    X_emb = _embedder.encode([txt])
    emb_pred = _clf_emb.predict(X_emb)[0]
    emb_prob = max(_clf_emb.predict_proba(X_emb)[0])
    emb_label = _label_encoder.inverse_transform([emb_pred])[0]

    # 3. Similarity (nearest neighbor)
    sims = cosine_similarity(X_emb, _embs)[0]
    sim_idx = int(np.argmax(sims))
    sim_label = _examples_y[sim_idx]
    sim_score = float(sims[sim_idx])

    # 4. Regra de decisão combinada
    # Se algum modelo tiver confiança > 0.75, retorna
    if rf_prob > 0.75:
        return rf_label, float(rf_prob)
    if emb_prob > 0.75:
        return emb_label, float(emb_prob)
    if sim_score > 0.65:
        return sim_label, float(sim_score)
    # Fallback: mais votado entre os três
    labels = [rf_label, emb_label, sim_label]
    voted = max(set(labels), key=labels.count)
    return voted, max(rf_prob, emb_prob, sim_score)

# Teste CLI
if __name__ == "__main__":
    features = {
        "tag": "SPAN",
        "class": "btn-primary",
        "id": "cta-main",
        "text": "Compre agora",
        "contextHeadings": ["Hero", "Ofertas"],
        "parents": [],
        "y": 8,
        "siblingIndex": 2
    }
    label, score = predict_session(features)
    print("Label:", label, "Score:", score)
