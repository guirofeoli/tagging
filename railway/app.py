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

@app.route("/", methods=["GET"])
def health():
    print("[Auto-UX] Healthcheck OK")
    return "Auto-UX API online", 200

@app.route("/get_examples", methods=["GET"])
def get_examples():
    print("[Auto-UX] GET /get_examples")
    content, sha = get_file_from_github(EXAMPLES_FILE)
    if not content:
        print("[Auto-UX] Nenhum exemplo no GitHub")
        return jsonify({"ok": True, "examples": []})
    try:
        data = json.loads(content)
    except Exception as e:
        print("[Auto-UX] ERRO ao ler exemplos do GitHub:", e)
        data = []
    return jsonify({"ok": True, "examples": data})

@app.route("/get_labels", methods=["GET"])
def get_labels():
    print("[Auto-UX] GET /get_labels")
    content, sha = get_file_from_github(LABELS_FILE)
    try:
        labels = json.loads(content) if content else []
        print(f"[Auto-UX] Labels retornados: {labels}")
        return jsonify({"ok": True if labels else False, "labels": labels})
    except Exception as e:
        print("[Auto-UX] ERRO ao ler labels:", e)
        return jsonify({"ok": False, "labels": []})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        from model_predict import predict_session
        features = request.get_json()
        print("[Auto-UX] Predição recebida. Features:", features)
        label, score = predict_session(features)
        print(f"[Auto-UX] Predição -> Label: {label} Score: {score}")
        return jsonify({"ok": True, "sessao": label, "score": score})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[Auto-UX] ERRO na predição:", e, tb)
        return jsonify({"ok": False, "msg": str(e), "traceback": tb}), 500

@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    print("[Auto-UX] POST /salvar_examples chamado")
    data = request.get_json()
    if not data or not isinstance(data, list):
        print("[Auto-UX] Payload inválido:", data)
        return jsonify({"ok": False, "msg": "Payload deve ser lista de exemplos"}), 400

    # Recupera exemplos existentes do GitHub e incrementa
    old_content, old_sha = get_file_from_github(EXAMPLES_FILE)
    exemplos_atuais = []
    if old_content:
        try:
            exemplos_atuais = json.loads(old_content)
        except Exception as e:
            print("[Auto-UX] ERRO ao decodificar exemplos antigos:", e)
            exemplos_atuais = []

    print(f"[Auto-UX] {len(exemplos_atuais)} exemplos antigos encontrados. {len(data)} novos exemplos recebidos.")
    exemplos_finais = exemplos_atuais + data

    # Salva no GitHub
    print(f"[Auto-UX] Salvando {len(exemplos_finais)} exemplos no GitHub...")
    status, resp = save_file_to_github(
        EXAMPLES_FILE,
        json.dumps(exemplos_finais, ensure_ascii=False, indent=2),
        "Incrementa exemplos UX",
        sha=old_sha
    )
    print(f"[Auto-UX] Status GitHub: {status}. Resposta: {str(resp)[:200]}")
    if status not in (200, 201):
        return jsonify({"ok": False, "msg": "Erro ao salvar exemplos no GitHub", "resp": resp}), 500

    # Treina modelo com todos exemplos agora salvos
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
        import traceback
        tb = traceback.format_exc()
        print("[Auto-UX] ERRO no treino automático:", e, tb)
        return jsonify({"ok": False, "msg": str(e), "traceback": tb}), 500

# --- CORS universal em toda resposta
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
    app.run(host="0.0.0.0", port=8080)
