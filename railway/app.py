import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os

# Funções do git_utils
from git_utils import get_file_from_github, save_file_to_github

app = Flask(__name__)
CORS(app, origins=["https://www.ton.com.br"], supports_credentials=True)

EXAMPLES_FILE = "ux_examples.json"
LABELS_FILE = "labels.json"
VOCAB_FILE = "allWords.json"

# --- Saudações simples
@app.route("/", methods=["GET"])
def health():
    return "Auto-UX API online", 200

# --- Retorna exemplos salvos no GitHub
@app.route("/get_examples", methods=["GET"])
def get_examples():
    content, sha = get_file_from_github(EXAMPLES_FILE)
    if not content:
        return jsonify({"ok": True, "examples": []})
    try:
        data = json.loads(content)
    except Exception:
        data = []
    return jsonify({"ok": True, "examples": data})

# --- Retorna labels.json do GitHub (ou vazio se não existir)
@app.route("/get_labels", methods=["GET"])
def get_labels():
    content, sha = get_file_from_github(LABELS_FILE)
    try:
        labels = json.loads(content) if content else []
        return jsonify({"ok": True if labels else False, "labels": labels})
    except Exception:
        return jsonify({"ok": False, "labels": []})

# --- Predição usando modelo treinado
@app.route("/predict", methods=["POST"])
def predict():
    try:
        from model_predict import predict_session
        features = request.get_json()
        label, score = predict_session(features)
        return jsonify({"ok": True, "sessao": label, "score": score})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

# --- Salva exemplos no GitHub e treina automaticamente
@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"ok": False, "msg": "Payload deve ser lista de exemplos"}), 400

    # 1. Recupera exemplos existentes do GitHub e incrementa (append)
    old_content, old_sha = get_file_from_github(EXAMPLES_FILE)
    exemplos_atuais = []
    if old_content:
        try:
            exemplos_atuais = json.loads(old_content)
        except Exception:
            exemplos_atuais = []

    # Junta antigos + novos
    exemplos_finais = exemplos_atuais + data

    # 2. Salva no GitHub (sempre commita!)
    status, resp = save_file_to_github(
        EXAMPLES_FILE,
        json.dumps(exemplos_finais, ensure_ascii=False, indent=2),
        "Incrementa exemplos UX",
        sha=old_sha
    )
    if status not in (200, 201):
        return jsonify({"ok": False, "msg": "Erro ao salvar exemplos no GitHub", "resp": resp}), 500

    # 3. Treina modelo com todos exemplos agora salvos
    try:
        import time
        print("[Auto-UX] Iniciando treinamento automático após salvar exemplos...")
        t0 = time.time()
        from train_ux import train_and_save_model
        # Salva temporário para treino local
        with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(exemplos_finais, f, ensure_ascii=False, indent=2)
        train_and_save_model(EXAMPLES_FILE)
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído em {t1-t0:.1f}s.")
        return jsonify({"ok": True, "msg": f"Incrementados {len(data)} exemplos. Modelo treinado!"})
    except Exception as e:
        print("[Auto-UX] ERRO no treino automático:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# --- CORS universal em toda resposta
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# --- Handler universal para OPTIONS (preflight)
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = make_response('', 200)
    response.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# --- CORS até em erros
@app.errorhandler(Exception)
def handle_error(e):
    import traceback
    tb = traceback.format_exc()
    resp = make_response(str(e) + "\n" + tb, 500)
    resp.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
