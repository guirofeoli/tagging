import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os

from git_utils import get_file_from_github, save_file_to_github

app = Flask(__name__)
CORS(app, origins=["https://www.ton.com.br"], supports_credentials=True)

EXAMPLES_FILE = "ux_examples.json"
LABELS_FILE = "labels.json"
VOCAB_FILE = "allWords.json"
MODEL_FILE = "model.bin"
USERS_FILE = "users.json"

@app.route("/", methods=["GET"])
def health():
    print("[Auto-UX] API online (healthcheck)")
    return "Auto-UX API online", 200

@app.route("/get_examples", methods=["GET"])
def get_examples():
    print("[Auto-UX] /get_examples chamado")
    content, sha = get_file_from_github(EXAMPLES_FILE)
    if not content:
        return jsonify({"ok": True, "examples": []})
    try:
        data = json.loads(content.decode("utf-8"))
    except Exception:
        data = []
    return jsonify({"ok": True, "examples": data})

@app.route("/get_labels", methods=["GET"])
def get_labels():
    print("[Auto-UX] /get_labels chamado")
    content, sha = get_file_from_github(LABELS_FILE)
    try:
        labels = json.loads(content.decode("utf-8")) if content else []
        print("[Auto-UX] Labels retornados (locais):", labels)
        return jsonify({"ok": True if labels else False, "labels": labels})
    except Exception:
        print("[Auto-UX] Erro ao carregar labels.json")
        return jsonify({"ok": False, "labels": []})

@app.route("/predict", methods=["POST"])
def predict():
    print("[Auto-UX] /predict chamado")
    try:
        from model_predict import predict_session
        features = request.get_json()
        print("[Auto-UX] Features recebidas:", features)
        label, score = predict_session(features)
        print("[Auto-UX] Predição:", label, score)
        return jsonify({"ok": True, "sessao": label, "score": score})
    except Exception as e:
        print("[Auto-UX] ERRO no predict:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    print("[Auto-UX] /salvar_examples CHAMADO")
    data = request.get_json()
    print("[Auto-UX] Dados recebidos (len):", len(data) if data else "Nada")
    if not data or not isinstance(data, list):
        print("[Auto-UX] Payload inválido:", data)
        return jsonify({"ok": False, "msg": "Payload deve ser lista de exemplos"}), 400

    old_content, old_sha = get_file_from_github(EXAMPLES_FILE)
    exemplos_atuais = []
    if old_content:
        try:
            exemplos_atuais = json.loads(old_content.decode("utf-8"))
        except Exception as ex:
            print("[Auto-UX] Falha ao ler exemplos antigos:", ex)
            exemplos_atuais = []

    exemplos_finais = exemplos_atuais + data
    print(f"[Auto-UX] Antes de salvar: antigos={len(exemplos_atuais)}, novos={len(data)}, final={len(exemplos_finais)}")

    status, resp = save_file_to_github(
        EXAMPLES_FILE,
        json.dumps(exemplos_finais, ensure_ascii=False, indent=2),
        "Incrementa exemplos UX",
        sha=old_sha
    )
    print("[Auto-UX] save_file_to_github status:", status, "resp:", str(resp)[:200])
    if status not in (200, 201):
        print("[Auto-UX] ERRO ao salvar no GitHub")
        return jsonify({"ok": False, "msg": "Erro ao salvar exemplos no GitHub", "resp": resp}), 500

    try:
        import time
        print("[Auto-UX] Iniciando treinamento automático após salvar exemplos...")
        t0 = time.time()
        from train_ux import train_and_save_model
        with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(exemplos_finais, f, ensure_ascii=False, indent=2)
        train_and_save_model(EXAMPLES_FILE)
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído em {t1-t0:.1f}s.")
        return jsonify({"ok": True, "msg": f"Incrementados {len(data)} exemplos. Modelo treinado!"})
    except Exception as e:
        print("[Auto-UX] ERRO no treino automático:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = make_response('', 200)
    response.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.errorhandler(Exception)
def handle_error(e):
    import traceback
    tb = traceback.format_exc()
    print("[Auto-UX] ERRO GLOBAL:", e, tb)
    resp = make_response(str(e) + "\n" + tb, 500)
    resp.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

if __name__ == "__main__":
    print("[Auto-UX] Iniciando API Railway")
    app.run(host="0.0.0.0", port=8080)
