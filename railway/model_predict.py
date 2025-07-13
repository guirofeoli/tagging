import json, joblib, os
import numpy as np

def extract_tokens(features):
    tokens = []
    for k in ['class', 'text', 'id', 'tag', 'selector', 'role', 'type', 'placeholder', 'href', 'title', 'alt']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'parents' in features:
        for p in features['parents']:
            for k in ['class', 'text', 'id', 'tag', 'selector']:
                val = p.get(k, "")
                if isinstance(val, str):
                    tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'contextHeadings' in features and features['contextHeadings']:
        for h in features['contextHeadings']:
            tokens += [w.lower() for w in h.split() if len(w) > 1]
    if 'y' in features: tokens.append(f'y{int(features['y']//10)*10}')
    if 'siblingIndex' in features: tokens.append(f'sib{min(int(features['siblingIndex']), 10)}')
    if 'width' in features: tokens.append(f'w{int(features['width']//50)*50}')
    if 'height' in features: tokens.append(f'h{int(features['height']//20)*20}')
    if 'depth' in features: tokens.append(f'd{min(int(features['depth']), 10)}')
    if 'clickable' in features: tokens.append(f'click{features['clickable']}')
    return tokens

def flatten_features(example):
    text_parts = [
        example.get('text',''), example.get('tag',''), example.get('id',''),
        example.get('class',''), example.get('selector',''),
        str(example.get('contextHeadings',[]))
    ]
    for p in example.get('parents',[]):
        text_parts.append(p.get('text',''))
        text_parts.append(p.get('selector',''))
    return " | ".join([str(x) for x in text_parts if x])

def predict_session(features):
    # --- Random Forest
    clf = joblib.load("model.bin")
    with open("allWords.json", encoding="utf-8") as f:
        vocab = json.load(f)
    X = [1 if w in set(extract_tokens(features)) else 0 for w in vocab]
    y_pred_rf = clf.predict([X])[0]
    rf_prob = max(clf.predict_proba([X])[0])

    # --- TF-IDF
    tfidf = joblib.load("model_tfidf.bin")
    with open("labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    with open("all_examples_texts.json", encoding="utf-8") as f:
        texts = json.load(f)
    f_text = flatten_features(features)
    vec = tfidf.transform([f_text])
    matrix = tfidf.transform(texts)
    sim_scores = np.dot(matrix, vec.T).toarray().flatten()
    idx_max = int(np.argmax(sim_scores))
    label_tfidf = None
    if sim_scores[idx_max] > 0.35:  # limiar ajustável
        label_tfidf = labels[idx_max]
        tfidf_prob = sim_scores[idx_max]
    else:
        label_tfidf = None
        tfidf_prob = 0

    # --- SBERT
    try:
        from sentence_transformers import SentenceTransformer
        with open("bert_emb_matrix.json", encoding="utf-8") as f:
            emb_matrix = np.array(json.load(f))
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        feat_emb = sbert_model.encode([f_text])[0]
        sims = np.dot(emb_matrix, feat_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(feat_emb) + 1e-8)
        idx_sbert = int(np.argmax(sims))
        sbert_label = labels[idx_sbert]
        sbert_prob = sims[idx_sbert]
    except Exception:
        sbert_label, sbert_prob = None, 0

    # --- Ensemble (votação)
    votes = {}
    for label, score in [(labels[y_pred_rf], rf_prob), (label_tfidf, tfidf_prob), (sbert_label, sbert_prob)]:
        if label:
            votes[label] = votes.get(label, 0) + score
    if votes:
        final_label = max(votes, key=votes.get)
        confidence = votes[final_label] / sum(votes.values())
        return final_label, float(confidence)
    # fallback: mais comum
    return labels[y_pred_rf], float(rf_prob)
