import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins=["https://www.ton.com.br"], supports_credentials=True)

def ensure_file(fname, default):
    if not os.path.exists(fname):
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False)
    return fname

EXAMPLES_FILE = ensure_file("ux_examples.json", [])
VOCAB_FILE = ensure_file("allWords.json", [])
LABELS_FILE = ensure_file("labels.json", [])

@app.route("/", methods=["GET"])
def health():
    return "Auto-UX API online", 200

@app.route("/get_examples", methods=["GET"])
def get_examples():
    with open(EXAMPLES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"ok": True, "examples": data})

@app.route("/get_labels", methods=["GET"])
def get_labels():
    try:
        if not os.path.exists(LABELS_FILE):
            return jsonify({"ok": False, "labels": []}), 200
        with open(LABELS_FILE, encoding="utf-8") as f:
            labels = json.load(f)
        return jsonify({"ok": True, "labels": labels})
    except Exception:
        return jsonify({"ok": False, "labels": []}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        from model_predict import predict_session
        features = request.get_json()
        label, score = predict_session(features)
        return jsonify({"ok": True, "sessao": label, "score": score})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"ok": False, "msg": "Payload deve ser lista de exemplos"}), 400
    # Salva exemplos
    with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    # Auto-treino
    try:
        import time
        print("[Auto-UX] Iniciando treinamento automático após salvar exemplos...")
        t0 = time.time()
        from train_ux import train_and_save_model
        train_and_save_model(EXAMPLES_FILE)
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído em {t1-t0:.1f}s.")
        return jsonify({"ok": True, "msg": f"Salvos {len(data)} exemplos e modelo treinado com sucesso!"})
    except Exception as e:
        print("[Auto-UX] ERRO no treino automático:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# -- CORS em todas as respostas --
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

# -- CORS até em erros 500/404/etc --
@app.errorhandler(Exception)
def handle_error(e):
    import traceback
    tb = traceback.format_exc()
    resp = make_response(str(e) + "\n" + tb, 500)
    resp.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
