# train_ux.py - atualizado para features enriquecidas
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Campos numéricos e de cor/estilo
NUMERIC_FIELDS = ["width", "height", "fontSize", "y", "siblingIndex"]
STYLE_FIELDS = ["bgColor", "fontWeight", "fontColor"]

def extract_tokens(features):
    """Extrai tokens categóricos do dict de features para one-hot."""
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
    # Novos campos de cor/estilo: simplificamos a cor
    for k in STYLE_FIELDS:
        if features.get(k):
            tokens.append(f"{k}={features[k]}")
    return tokens

def build_vocab(examples):
    vocab = set()
    for ex in examples:
        tokens = extract_tokens(ex)
        vocab.update(tokens)
    return sorted(vocab)

def example_to_vector(example, vocab, numeric_means=None, numeric_stds=None):
    # Vetor one-hot para categorias/texto
    tokens = set(extract_tokens(example))
    vector = [1 if w in tokens else 0 for w in vocab]
    # Vetor para campos numéricos (normalizado)
    for f in NUMERIC_FIELDS:
        val = float(example.get(f, 0) or 0)
        # Normalização Z-score, se fornecido
        if numeric_means and numeric_stds and f in numeric_means:
            std = numeric_stds[f] if numeric_stds[f] > 0 else 1
            val = (val - numeric_means[f]) / std
        vector.append(val)
    return vector

def compute_numeric_stats(examples):
    vals = {f: [] for f in NUMERIC_FIELDS}
    for ex in examples:
        for f in NUMERIC_FIELDS:
            try:
                vals[f].append(float(ex.get(f, 0) or 0))
            except:
                vals[f].append(0)
    means = {f: np.mean(vals[f]) for f in NUMERIC_FIELDS}
    stds  = {f: np.std(vals[f]) for f in NUMERIC_FIELDS}
    return means, stds

def train_and_save_model(json_path="ux_examples.json"):
    print("[Auto-UX] Lendo exemplos rotulados...")
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise Exception("Nenhum exemplo rotulado encontrado!")
    print(f"[Auto-UX] {len(examples)} exemplos carregados.")

    vocab = build_vocab(examples)
    print(f"[Auto-UX] Vocab extraído: {len(vocab)} tokens.")

    # Calcula stats para normalizar campos numéricos
    numeric_means, numeric_stds = compute_numeric_stats(examples)

    X = [example_to_vector(ex, vocab, numeric_means, numeric_stds) for ex in examples]
    y_raw = [ex["sessao"] for ex in examples]

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("[Auto-UX] Treinando modelo RandomForest...")
    clf = RandomForestClassifier(n_estimators=140, random_state=42)
    clf.fit(X, y)
    print("[Auto-UX] Modelo treinado!")

    # Salva tudo que for preciso para predição
    joblib.dump(clf, "model.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    with open("numeric_means.json", "w", encoding="utf-8") as f:
        json.dump(numeric_means, f, ensure_ascii=False)
    with open("numeric_stds.json", "w", encoding="utf-8") as f:
        json.dump(numeric_stds, f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, stats).")

if __name__ == "__main__":
    train_and_save_model()
