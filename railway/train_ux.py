import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from git_utils import save_file_to_github, get_file_from_github
import base64

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

    # Estatísticas numéricas extras (exemplo: média dos numéricos para normalização futura)
    numeric_means = {}
    numerics = ["length", "y", "siblingIndex", "width", "height", "depth", "numChildren"]
    for n in numerics:
        vals = [ex.get(n, 0) for ex in examples if isinstance(ex.get(n, 0), (int, float))]
        if vals:
            numeric_means[n] = float(np.mean(vals))
    with open("numeric_means.json", "w", encoding="utf-8") as f:
        json.dump(numeric_means, f, ensure_ascii=False)

    # Salva o modelo e os arquivos auxiliares
    joblib.dump(clf, "model.bin")
    with open("allWords.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, stats).")

    # Agora, exporta para o GitHub
    files = [
        ("model.bin", "rb", True),
        ("allWords.json", "r", False),
        ("labels.json", "r", False),
        ("numeric_means.json", "r", False)
    ]
    for fname, mode, is_binary in files:
        try:
            old_content, old_sha = get_file_from_github(fname)
            if is_binary:
                with open(fname, "rb") as f:
                    content = base64.b64encode(f.read()).decode("utf-8")
            else:
                with open(fname, "r", encoding="utf-8") as f:
                    content = f.read()
            status, resp = save_file_to_github(fname, content, "Atualiza "+fname, sha=old_sha, is_binary=is_binary)
            print(f"[Auto-UX] {fname} save_file_to_github: status {status}")
        except Exception as e:
            print(f"[Auto-UX] Falha ao exportar {fname}:", e)

if __name__ == "__main__":
    train_and_save_model()
