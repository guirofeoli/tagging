# app.py
from flask import Flask, request, jsonify
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Endereços
HOSTGATOR_UX_EXAMPLES = "https://rofeoli.com/ux-model/ux_examples.json"
UPLOAD_URL = "https://rofeoli.com/ux-model/upload.php"
UPLOAD_TOKEN = "Rofeoli@gui1"  # Altere aqui se mudar o token

# Carrega o modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/train', methods=['POST'])
def train():
    # 1. Baixar os exemplos rotulados
    r = requests.get(HOSTGATOR_UX_EXAMPLES)
    data = r.json()

    # 2. Gera embeddings dos textos + contextos, pega os labels
    texts = []
    labels = []
    for ex in data:
        ctx = ' '.join(ex.get('contextHeadings', []))
        fulltext = f"{ex.get('text','')} {ctx} {ex.get('class','')} {ex.get('id','')}"
        texts.append(fulltext.strip())
        labels.append(ex['sessao'])
    embs = model.encode(texts)

    # 3. Salva localmente (Railway) para upload
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embs.tolist(), f)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    # 4. Envia pro HostGator via upload.php
    # Envia embeddings.json
    files = {'file': ('embeddings.json', open('embeddings.json', 'rb'), 'application/json')}
    r1 = requests.post(UPLOAD_URL, files=files, data={'token': UPLOAD_TOKEN})
    # Envia labels.json
    files = {'file': ('labels.json', open('labels.json', 'rb'), 'application/json')}
    r2 = requests.post(UPLOAD_URL, files=files, data={'token': UPLOAD_TOKEN})

    return jsonify({
        "ok": True,
        "msg": "Arquivos enviados para HostGator!",
        "embeddings_upload": r1.text,
        "labels_upload": r2.text,
        "num_examples": len(texts)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
