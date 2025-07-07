import os
import json
from flask import Flask, request, jsonify, send_file
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Locais dos arquivos
EMBEDDINGS_FILE = "embeddings.npy"
LABELS_FILE = "labels.json"

# Model global para não recarregar a cada chamada
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/")
def home():
    return "<h3>Auto-UX API online 🚀</h3>"

@app.route("/treinar", methods=["POST"])
def treinar():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"erro": "Payload deve ser um array de exemplos!"}), 400

    texts = [str(e.get("text", "")) for e in data]
    labels = sorted(list(set(e.get("sessao", "") for e in data)))
    X = model.encode(texts)
    y = np.array([labels.index(e["sessao"]) for e in data])

    np.save(EMBEDDINGS_FILE, X)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    return jsonify({"msg": f"Treinamento concluído! {len(labels)} labels, {len(X)} exemplos."})

@app.route("/embeddings.npy")
def download_embeddings():
    return send_file(EMBEDDINGS_FILE, as_attachment=True)

@app.route("/labels.json")
def download_labels():
    return send_file(LABELS_FILE, as_attachment=True)

# (Opcional) — endpoint de predição
@app.route("/predizer", methods=["POST"])
def predizer():
    d = request.get_json()
    if not d or "text" not in d:
        return jsonify({"erro": "Envie {'text': ...}"}), 400
    with open(LABELS_FILE, encoding="utf-8") as f:
        labels = json.load(f)
    X = model.encode([d["text"]])
    embs = np.load(EMBEDDINGS_FILE)
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(X, embs)[0]
    idx = np.argmax(scores)
    return jsonify({
        "sessao_predita": labels[idx],
        "score": float(scores[idx])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
