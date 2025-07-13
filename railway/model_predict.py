# model_predict.py
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import base64, tarfile, io, os, shutil

_model = None
_vocab = None
_labels = None
_tfidf_model = None
_all_texts = None
_bert_emb_matrix = None
_sbert_model = None

def flatten_features(example):
    text_parts = [
        example.get('text',''),
        example.get('tag',''),
        example.get('id',''),
        example.get('class',''),
        example.get('selector',''),
        str(example.get('contextHeadings',[])),
    ]
    for p in example.get('parents',[]):
        text_parts.append(p.get('text',''))
        text_parts.append(p.get('selector',''))
    return " | ".join([str(x) for x in text_parts if x])

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
    if 'y' in features: tokens.append(f'y{int(features["y"]//10)*10}')
    if 'siblingIndex' in features: tokens.append(f'sib{min(int(features["siblingIndex"]), 10)}')
    if 'width' in features: tokens.append(f'w{int(features["width"]//50)*50}')
    if 'height' in features: tokens.append(f'h{int(features["height"]//20)*20}')
    if 'depth' in features: tokens.append(f'd{min(int(features["depth"]), 10)}')
    if 'clickable' in features: tokens.append(f'click{features["clickable"]}')
    return tokens

def build_vocab(examples):
    vocab = set()
    for ex in examples:
        tokens = extract_tokens(ex)
        vocab.update(tokens)
    return sorted(vocab)

def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

def load_sbert_base64(filename):
    global _sbert_model
    with open(filename, "r", encoding="utf-8") as f:
        tar_b64 = json.load(f)["base64"]
        tar_bytes = base64.b64decode(tar_b64)
        tmpdir = "./sbert_tmp_model"
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(tmpdir)
        _sbert_model = SentenceTransformer(tmpdir + "/model")
    return _sbert_model

def _load_all():
    global _model, _vocab, _labels, _tfidf_model, _all_texts, _bert_emb_matrix, _sbert_model
    if _model is None:
        _model = joblib.load("model.bin")
    if _vocab is None:
        with open("allWords.json", "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _labels is None:
        with open("labels.json", "r", encoding="utf-8") as f:
            _labels = json.load(f)
    if _tfidf_model is None:
        _tfidf_model = joblib.load("model_tfidf.bin")
    if _all_texts is None:
        with open("all_examples_texts.json", "r", encoding="utf-8") as f:
            _all_texts = json.load(f)
    if _bert_emb_matrix is None:
        with open("bert_emb_matrix.json", "r", encoding="utf-8") as f:
            _bert_emb_matrix = np.array(json.load(f))
    if _sbert_model is None:
        load_sbert_base64("sbert_model_b64.json")

def predict_session(features):
    _load_all()
    # RandomForest
    X = np.array([example_to_vector(features, _vocab)])
    probs = _model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    # Defensive check:
    if not _labels:
        print("[Auto-UX] WARNING: labels.json vazio, retornando 'Não rotulado'")
        return "Não rotulado", 0.0
    if idx >= len(_labels):
        print(f"[Auto-UX] WARNING: Índice de predição RF ({idx}) fora do range do labels ({len(_labels)})")
        label = _labels[0]
        score = float(probs[0])
    else:
        label = _labels[idx]
        score = float(probs[idx])

    # SBERT
    sbert_vec = _sbert_model.encode([flatten_features(features)])[0]
    sims_bert = cosine_similarity([sbert_vec], _bert_emb_matrix)[0]
    idx_bert = int(np.argmax(sims_bert))
    if idx_bert >= len(_labels):
        print(f"[Auto-UX] WARNING: Índice SBERT ({idx_bert}) fora do range dos labels ({len(_labels)})")
        sim_label = _labels[0]
        sim_score = float(sims_bert[0])
    else:
        sim_label = _labels[idx_bert]
        sim_score = float(sims_bert[idx_bert])

    # TF-IDF Fallback
    tfidf_test = _tfidf_model.transform([flatten_features(features)])
    tfidf_examples = _tfidf_model.transform(_all_texts)
    sims = cosine_similarity(tfidf_test, tfidf_examples)[0]
    idx_tfidf = int(np.argmax(sims))
    if idx_tfidf >= len(_labels):
        print(f"[Auto-UX] WARNING: Índice TFIDF ({idx_tfidf}) fora do range dos labels ({len(_labels)})")
        label_tfidf = _labels[0]
        sim_score_tfidf = float(sims[0])
    else:
        label_tfidf = _labels[idx_tfidf]
        sim_score_tfidf = float(sims[idx_tfidf])

    # Decision logic
    if score >= 0.6:
        return label, score
    elif sim_score >= 0.60:
        return sim_label, sim_score
    else:
        return label_tfidf, sim_score_tfidf

# Teste manual:
if __name__ == "__main__":
    example = {
        "tag": "A",
        "class": "menu principal",
        "id": "main-menu",
        "text": "Início",
        "parents": [{"tag": "NAV", "class": "nav", "id": "nav-main", "text": "Menu principal"}],
        "contextHeadings": ["Menu"],
        "y": 9,
        "siblingIndex": 0,
        "selector": "nav.nav > ul > li"
    }
    label, score = predict_session(example)
    print("Label:", label, "Score:", score)
