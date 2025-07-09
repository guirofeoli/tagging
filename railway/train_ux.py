# train_ux.py
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    from git_utils import save_file_to_github
except ImportError:
    # Para teste local, ignora upload ao GitHub
    def save_file_to_github(*a, **k): return (200, {"dummy": True})

EXAMPLES_FILE = "ux_examples.json"
MODEL_FILE = "model.bin"
VOCAB_FILE = "allWords.json"
LABELS_FILE = "labels.json"
NUMERIC_MEANS_FILE = "numeric_means.json"

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

def example_to_vector(example, vocab, numeric_keys=None, numeric_means=None):
    tokens = set(extract_tokens(example))
    # One-hot textual
    vector = [1 if w in tokens else 0 for w in vocab]
    # Campos numéricos enriquecidos (padronizados)
    if numeric_keys:
        for k in numeric_keys:
            val = example.get(k, None)
            if val is None:
                val = numeric_means.get(k, 0) if numeric_means else 0
            vector.append(float(val))
    return vector

def train_and_save_model(json_path=EXAMPLES_FILE):
    print("[Auto-UX] Lendo exemplos rotulados...")
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        raise Exception("Nenhum exemplo rotulado encontrado!")
    print(f"[Auto-UX] {len(examples)} exemplos carregados.")

    # Defina os campos numéricos adicionais que deseja usar:
    numeric_keys = [
        "length", "y", "siblingIndex", "width", "height", "visible", "depth", "numChildren"
    ]
    # Calcule médias dos campos numéricos para preencher valores faltantes
    numeric_means = {}
    for k in numeric_keys:
        arr = [ex.get(k, 0) for ex in examples if k in ex]
        numeric_means[k] = float(np.mean(arr)) if arr else 0

    # Salva médias para usar no predict
    with open(NUMERIC_MEANS_FILE, "w", encoding="utf-8") as f:
        json.dump(numeric_means, f, ensure_ascii=False)
    print("[Auto-UX] Média dos campos numéricos salva em numeric_means.json.")

    # Salva no GitHub (Railway)
    status, resp = save_file_to_github(NUMERIC_MEANS_FILE, json.dumps(numeric_means, ensure_ascii=False, indent=2), "Atualiza numeric_means.json", sha=None)
    print("[Auto-UX] numeric_means.json save_file_to_github:", status)

    # Monta vocab e extrai features
    vocab = build_vocab(examples)
    print(f"[Auto-UX] Vocab extraído: {len(vocab)} tokens.")

    X = [example_to_vector(ex, vocab, numeric_keys, numeric_means) for ex in examples]
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
    joblib.dump(clf, MODEL_FILE)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, stats).")

    # Salva no GitHub
    for fname, fcontent in [
        (LABELS_FILE, json.dumps(list(label_encoder.classes_), ensure_ascii=False, indent=2)),
        (VOCAB_FILE, json.dumps(vocab, ensure_ascii=False, indent=2)),
    ]:
        status, resp = save_file_to_github(fname, fcontent, f"Atualiza {fname}", sha=None)
        print(f"[Auto-UX] {fname} save_file_to_github: status {status}")

if __name__ == "__main__":
    train_and_save_model()
