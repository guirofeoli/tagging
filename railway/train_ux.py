import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib

# Load data
with open("ux_examples.json", encoding="utf-8") as f:
    examples = json.load(f)

# Função para extrair tokens clássicos
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
        tokens.append(f'y{int(features['y']//10)*10}')
    if 'siblingIndex' in features:
        tokens.append(f'sib{min(int(features['siblingIndex']), 10)}')
    return tokens

# Build vocabulary
def build_vocab(examples):
    vocab = set()
    for ex in examples:
        tokens = extract_tokens(ex)
        vocab.update(tokens)
    return sorted(vocab)

# Token -> one-hot vector
def example_to_vector(example, vocab):
    tokens = set(extract_tokens(example))
    return [1 if w in tokens else 0 for w in vocab]

# Text for embedding
def features_to_text(features):
    txt = [
        features.get('text', ''),
        features.get('class', ''),
        features.get('id', ''),
        features.get('tag', '')
    ] + features.get('contextHeadings', [])
    return " ".join([t for t in txt if t])

# Preparação
vocab = build_vocab(examples)
X_rf = [example_to_vector(ex, vocab) for ex in examples]
texts = [features_to_text(ex) for ex in examples]
y_raw = [ex["sessao"] for ex in examples]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_emb = embedder.encode(texts, show_progress_bar=True)

# Treina Random Forest (tokens)
rf = RandomForestClassifier(n_estimators=120, random_state=42)
rf.fit(X_rf, y)

# Treina Logistic Regression (embeddings)
clf_emb = LogisticRegression(max_iter=400)
clf_emb.fit(X_emb, y)

# Salva modelos e auxiliares
joblib.dump(rf, "rf_model.bin")
joblib.dump(clf_emb, "emb_model.bin")
joblib.dump(label_encoder, "label_encoder.bin")
np.save("embeddings.npy", X_emb)
with open("allWords.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(list(label_encoder.classes_), f, ensure_ascii=False)
with open("examples_meta.json", "w", encoding="utf-8") as f:
    json.dump(y_raw, f, ensure_ascii=False)
print("Tudo salvo! Modelos e auxiliares prontos.")
