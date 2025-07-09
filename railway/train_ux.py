# train_ux.py
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from git_utils import save_file_to_github, get_file_from_github
import base64

def extract_tokens(features):
    """Extrai tokens simples do dicionário de features para o one-hot."""
    tokens = []
    for k in ['class', 'text', 'id', 'tag']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    # Pais e headings
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

def build_vocab(examples):
    vocab = set()
    for ex in examples:
        tokens = extract_tokens(ex)
        vocab.update(tokens)
    return sorted(vocab)

def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

def train_and_save_model(json_path="ux_examples.json"):
    print("[Auto-UX] Lendo exemplos rotulados...")
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise Exception("Nenhum exemplo rotulado encontrado!")
    print(f"[Auto-UX] {len(examples)} exemplos carregados.")

    # Monta vocab e extrai features
    vocab = build_vocab(examples)
    print(f"[Auto-UX] Vocab extraído: {len(vocab)} tokens.")
    X = [example_to_vector(ex, vocab) for ex in examples]
    y_raw = [ex["sessao"] for ex in examples]

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Treina o modelo
    print("[Auto-UX] Treinando modelo RandomForest...")
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y)
    print("[Auto-UX] Modelo treinado!")

    # Salva o modelo e os arquivos auxiliares localmente
    joblib.dump(clf, "model.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    print("[Auto-UX] Modelo salvo em model.bin, vocab salvo em allWords.json, labels em labels.json.")

    # ------- Envia para o GitHub -------
    print("[Auto-UX] Enviando arquivos de modelo para o GitHub...")

    # model.bin
    with open("model.bin", "rb") as f:
        model_content = base64.b64encode(f.read()).decode("utf-8")
    _, sha_model = get_file_from_github("model.bin")
    status_model, resp_model = save_file_to_github("model.bin", model_content, "Atualiza modelo treinado", sha=sha_model)
    print(f"[Auto-UX] model.bin GitHub status: {status_model}")

    # allWords.json
    with open("allWords.json", "r", encoding="utf-8") as f:
        vocab_content = f.read()
    _, sha_vocab = get_file_from_github("allWords.json")
    status_vocab, resp_vocab = save_file_to_github("allWords.json", vocab_content, "Atualiza vocab", sha=sha_vocab)
    print(f"[Auto-UX] allWords.json GitHub status: {status_vocab}")

    # labels.json
    with open("labels.json", "r", encoding="utf-8") as f:
        labels_content = f.read()
    _, sha_labels = get_file_from_github("labels.json")
    status_labels, resp_labels = save_file_to_github("labels.json", labels_content, "Atualiza labels", sha=sha_labels)
    print(f"[Auto-UX] labels.json GitHub status: {status_labels}")

    print("[Auto-UX] Upload para o GitHub finalizado.")

if __name__ == "__main__":
    train_and_save_model()
