# train_ux.py
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

def upload_artifacts_to_github(artifacts):
    # Upload de todos artefatos relevantes (textos como str, binários como base64 e is_binary=True)
    for fname, meta in artifacts.items():
        try:
            print(f"[Auto-UX] Enviando {fname} para o GitHub...")
            old_content, old_sha = get_file_from_github(fname)
            content = meta['content']
            commit_msg = meta.get('commit_msg', f"Atualiza {fname}")
            is_binary = meta.get('is_binary', False)
            # Se for bytes, sempre envia como binário/base64
            if isinstance(content, bytes):
                content = base64.b64encode(content).decode("utf-8")
                is_binary = True
            status, resp = save_file_to_github(
                fname, content, commit_msg, sha=old_sha, is_binary=is_binary
            )
            print(f"[Auto-UX] Upload {fname}: status {status}")
            if status not in (200, 201):
                print(f"[Auto-UX] Falha ao enviar {fname}: {resp}")
        except Exception as e:
            print(f"[Auto-UX] ERRO ao enviar {fname} para o GitHub: {e}")

def train_and_save_model(json_path="ux_examples.json"):
    print("[Auto-UX] Lendo exemplos rotulados...")
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise Exception("Nenhum exemplo rotulado encontrado!")
    print(f"[Auto-UX] {len(examples)} exemplos carregados.")

    # Remove duplicados (caso existam)
    seen = set()
    uniques = []
    for ex in examples:
        as_str = json.dumps(ex, sort_keys=True)
        if as_str not in seen:
            seen.add(as_str)
            uniques.append(ex)
    examples = uniques
    print(f"[Auto-UX] {len(examples)} exemplos após remover duplicados.")

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

    # TF-IDF para similaridade textual
    texts = [flatten_features(ex) for ex in examples]
    tfidf = TfidfVectorizer().fit(texts)
    print("[Auto-UX] TF-IDF treinado!")

    # SBERT embeddings
    print("[Auto-UX] Gerando embeddings BERT...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_matrix = sbert_model.encode(texts, show_progress_bar=False)
    with open("bert_emb_matrix.json", "w", encoding="utf-8") as f:
        json.dump(emb_matrix.tolist(), f)

    # Salva SBERT como base64 tar.gz (como string base64)
    with tempfile.TemporaryDirectory() as tmpdir:
        sbert_model.save(tmpdir)
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode='w:gz') as tar:
            tar.add(tmpdir, arcname="model")
        tar_b64 = base64.b64encode(tar_bytes.getvalue()).decode()
        with open("sbert_model_b64.json", "w", encoding="utf-8") as f:
            json.dump({"base64": tar_b64}, f)

    # Salva artefatos localmente
    joblib.dump(clf, "model.bin")
    joblib.dump(tfidf, "model_tfidf.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    with open("all_examples_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, tfidf, textos, embeddings, sbert-model)!")

    # Prepara conteúdo para upload no GitHub
    artifacts = {
        "model.bin": {"content": open("model.bin", "rb").read(), "is_binary": True},
        "model_tfidf.bin": {"content": open("model_tfidf.bin", "rb").read(), "is_binary": True},
        "allWords.json": {"content": open("allWords.json", "r", encoding="utf-8").read(), "is_binary": False},
        "labels.json": {"content": open("labels.json", "r", encoding="utf-8").read(), "is_binary": False},
        "all_examples_texts.json": {"content": open("all_examples_texts.json", "r", encoding="utf-8").read(), "is_binary": False},
        "bert_emb_matrix.json": {"content": open("bert_emb_matrix.json", "r", encoding="utf-8").read(), "is_binary": False},
        "sbert_model_b64.json": {"content": open("sbert_model_b64.json", "r", encoding="utf-8").read(), "is_binary": False},
    }
    upload_artifacts_to_github(artifacts)

if __name__ == "__main__":
    train_and_save_model()
