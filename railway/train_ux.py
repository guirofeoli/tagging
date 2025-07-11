# train_ux.py
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def extract_tokens(features):
    """Extrai tokens de todos os campos textuais e selectors para o one-hot."""
    tokens = []
    for k in ['class', 'text', 'id', 'tag', 'selector']:
        val = features.get(k, "")
        if isinstance(val, str):
            tokens += [w.lower() for w in val.split() if len(w) > 1]
    # Pais e headings
    if 'parents' in features:
        for p in features['parents']:
            for k in ['class', 'text', 'id', 'tag', 'selector']:
                val = p.get(k, "")
                if isinstance(val, str):
                    tokens += [w.lower() for w in val.split() if len(w) > 1]
    if 'contextHeadings' in features and features['contextHeadings']:
        for h in features['contextHeadings']:
            tokens += [w.lower() for w in h.split() if len(w) > 1]
    # Numéricas discretizadas
    if 'y' in features:
        tokens.append(f'y{int(features["y"]//10)*10}')
    if 'siblingIndex' in features:
        tokens.append(f'sib{min(int(features["siblingIndex"]), 10)}')
    if 'depth' in features:
        tokens.append(f'depth{min(int(features["depth"]), 15)}')
    if 'visible' in features:
        tokens.append(f'vis{int(features["visible"])}')
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

def compute_numeric_means(examples):
    # Média dos valores para normalização futura
    keys = ['length', 'y', 'width', 'height', 'depth', 'numChildren']
    means = {}
    for k in keys:
        vals = [float(ex.get(k, 0)) for ex in examples if k in ex]
        means[k] = float(np.mean(vals)) if vals else 0.0
    return means

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

    # Salva o modelo e os arquivos auxiliares
    joblib.dump(clf, "model.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    with open("numeric_means.json", "w", encoding="utf-8") as f:
        json.dump(compute_numeric_means(examples), f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, stats).")

if __name__ == "__main__":
    train_and_save_model()
