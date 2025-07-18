import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sentence_transformers import SentenceTransformer
import tempfile, shutil, tarfile, io, base64

from git_utils import save_file_to_github, get_file_from_github

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
    if 'y' in features: tokens.append(f'y{int(features['y']//10)*10}')
    if 'siblingIndex' in features: tokens.append(f'sib{min(int(features['siblingIndex']), 10)}')
    if 'width' in features: tokens.append(f'w{int(features['width']//50)*50}')
    if 'height' in features: tokens.append(f'h{int(features['height']//20)*20}')
    if 'depth' in features: tokens.append(f'd{min(int(features['depth']), 10)}')
    if 'clickable' in features: tokens.append(f'click{features['clickable']}')
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

def upload_artifacts_to_github(artifacts):
    for fname, (filedata, is_binary) in artifacts.items():
        # NUNCA subir o sbert_model_b64.json (muito grande, usado só local)
        if fname == "sbert_model_b64.json":
            print(f"[Auto-UX] Pulando upload de {fname} (muito grande, só local)")
            continue
        if is_binary and isinstance(filedata, bytes) and len(filedata) > 90*1024*1024:
            print(f"[Auto-UX] Pulando upload de {fname}: arquivo > 90MB.")
            continue
        print(f"[Auto-UX] Enviando {fname} para o GitHub...")
        _, sha = get_file_from_github(fname)
        status, resp = save_file_to_github(fname, filedata, "Atualiza " + fname, sha=sha, is_binary=is_binary)
        print(f"[Auto-UX] Upload {fname}: status {status}")

def train_and_save_model(json_path="ux_examples.json"):
    print("[Auto-UX] Lendo exemplos rotulados...")
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise Exception("Nenhum exemplo rotulado encontrado!")
    print(f"[Auto-UX] {len(examples)} exemplos carregados.")

    # Remove duplicados
    seen = set()
    filtered = []
    for ex in examples:
        sig = json.dumps(ex, sort_keys=True)
        if sig not in seen:
            filtered.append(ex)
            seen.add(sig)
    examples = filtered
    print(f"[Auto-UX] {len(examples)} exemplos após remover duplicados.")

    # RandomForest
    vocab = build_vocab(examples)
    print(f"[Auto-UX] Vocab extraído: {len(vocab)} tokens.")
    X = [example_to_vector(ex, vocab) for ex in examples]
    y_raw = [ex["sessao"] for ex in examples]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print("[Auto-UX] Treinando modelo RandomForest...")
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y)
    print("[Auto-UX] Modelo RF treinado!")

    # TF-IDF
    texts = [flatten_features(ex) for ex in examples]
    tfidf = TfidfVectorizer().fit(texts)
    print("[Auto-UX] TF-IDF treinado!")

    # SBERT embeddings
    print("[Auto-UX] Gerando embeddings BERT...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_matrix = sbert_model.encode(texts, show_progress_bar=False)
    with open("bert_emb_matrix.json", "w", encoding="utf-8") as f:
        json.dump(emb_matrix.tolist(), f)

    # Salva SBERT como base64 tar.gz só local
    with tempfile.TemporaryDirectory() as tmpdir:
        sbert_model.save(tmpdir)
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode='w:gz') as tar:
            tar.add(tmpdir, arcname="model")
        tar_b64 = base64.b64encode(tar_bytes.getvalue()).decode()
        with open("sbert_model_b64.json", "w", encoding="utf-8") as f:
            json.dump({"base64": tar_b64}, f)

    # Persiste tudo
    joblib.dump(clf, "model.bin")
    joblib.dump(tfidf, "model_tfidf.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    with open("all_examples_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, tfidf, textos, embeddings, sbert-model)!")

    # Upload para o GitHub (exceto SBERT b64)
    artifacts = {
        "model.bin": (open("model.bin", "rb").read(), True),
        "model_tfidf.bin": (open("model_tfidf.bin", "rb").read(), True),
        "allWords.json": (open("allWords.json", "r").read(), False),
        "labels.json": (open("labels.json", "r").read(), False),
        "all_examples_texts.json": (open("all_examples_texts.json", "r").read(), False),
        "bert_emb_matrix.json": (open("bert_emb_matrix.json", "r").read(), False),
        # "sbert_model_b64.json": (open("sbert_model_b64.json", "r").read(), False),  # nunca sobe!
    }
    upload_artifacts_to_github(artifacts)

if __name__ == "__main__":
    train_and_save_model()
