import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Funções do git_utils
from git_utils import save_file_to_github, get_file_from_github

EXAMPLES_FILE = "ux_examples.json"
VOCAB_FILE = "allWords.json"
LABELS_FILE = "labels.json"
MODEL_FILE = "model.bin"
NUMERIC_MEANS_FILE = "numeric_means.json"  # se usar features numéricas

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

def train_and_save_model(json_path=EXAMPLES_FILE):
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
    joblib.dump(clf, MODEL_FILE)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(label_encoder.classes_), f, ensure_ascii=False)

    # Se quiser calcular médias para features numéricas (opcional)
    numeric_means = None
    try:
        numeric_features = []
        for ex in examples:
            feats = []
            for k in ["length", "y", "siblingIndex", "width", "height", "visible", "depth", "numChildren"]:
                feats.append(ex.get(k, 0))
            numeric_features.append(feats)
        arr = np.array(numeric_features)
        numeric_means = dict(zip(
            ["length", "y", "siblingIndex", "width", "height", "visible", "depth", "numChildren"],
            np.nanmean(arr, axis=0)
        ))
        with open(NUMERIC_MEANS_FILE, "w", encoding="utf-8") as f:
            json.dump(numeric_means, f, ensure_ascii=False)
    except Exception as e:
        print("[Auto-UX] (Opcional) Falha ao calcular médias numéricas:", e)

    # Salva todos os arquivos também no GitHub!
    print("[Auto-UX] Salvando arquivos do modelo no GitHub...")
    # Upload model.bin (binário!)
    with open(MODEL_FILE, "rb") as f:
        save_file_to_github(MODEL_FILE, f.read(), "Atualiza modelo treinado UX", is_binary=True)
    # Upload vocab
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        save_file_to_github(VOCAB_FILE, f.read(), "Atualiza vocab UX")
    # Upload labels
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        save_file_to_github(LABELS_FILE, f.read(), "Atualiza labels treinados UX")
    # Upload numeric_means.json se existir
    if numeric_means is not None:
        with open(NUMERIC_MEANS_FILE, "r", encoding="utf-8") as f:
            save_file_to_github(NUMERIC_MEANS_FILE, f.read(), "Atualiza numeric_means UX")

    print("[Auto-UX] Tudo salvo (modelo, vocab, labels, stats).")

if __name__ == "__main__":
    train_and_save_model()
