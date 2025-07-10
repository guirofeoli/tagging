import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import time

from git_utils import get_file_from_github, save_file_to_github

app = Flask(__name__)
CORS(app, origins=["https://www.ton.com.br"], supports_credentials=True)

EXAMPLES_FILE = "ux_examples.json"
LABELS_FILE = "labels.json"
VOCAB_FILE = "allWords.json"
MODEL_FILE = "model.bin"
USERS_FILE = "users.json"

def sync_from_github():
    """Baixa todos os arquivos essenciais do GitHub se não existem localmente."""
    for fname in [EXAMPLES_FILE, LABELS_FILE, VOCAB_FILE, MODEL_FILE]:
        if not os.path.exists(fname):
            content, _ = get_file_from_github(fname)
            if content and fname != MODEL_FILE:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(content)
            elif content and fname == MODEL_FILE:
                # model.bin é binário!
                import base64
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(content))

@app.route("/", methods=["GET"])
def health():
    return "Auto-UX API online", 200

@app.route("/debug_ls", methods=["GET"])
def debug_ls():
    base = "."
    out = []
    for root, dirs, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            out.append(p)
    return jsonify(out)

@app.route("/debug_file", methods=["GET"])
def debug_file():
    fname = request.args.get("file")
    login = request.args.get("login", "")
    senha = request.args.get("senha", "")
    if not fname or ".." in fname:
        return "Arquivo não permitido", 400

    try:
        with open(USERS_FILE, encoding="utf-8") as f:
            users = json.load(f)
        valid = any(u.get("login") == login and u.get("senha") == senha for u in users)
        if not valid:
            return "Acesso negado!", 401
    except Exception as e:
        return f"Erro ao acessar users.json: {e}", 500

    try:
        with open(fname, encoding="utf-8") as f:
            conteudo = f.read()
        return "<pre>" + conteudo.replace("<", "&lt;") + "</pre>"
    except Exception as e:
        return f"Erro ao ler arquivo: {e}", 500

@app.route("/get_examples", methods=["GET"])
def get_examples():
    sync_from_github()
    content, sha = get_file_from_github(EXAMPLES_FILE)
    if not content:
        return jsonify({"ok": True, "examples": []})
    try:
        data = json.loads(content)
    except Exception:
        data = []
    return jsonify({"ok": True, "examples": data})

@app.route("/get_labels", methods=["GET"])
def get_labels():
    sync_from_github()
    content, sha = get_file_from_github(LABELS_FILE)
    try:
        labels = json.loads(content) if content else []
        return jsonify({"ok": True if labels else False, "labels": labels})
    except Exception:
        return jsonify({"ok": False, "labels": []})

@app.route("/predict", methods=["POST"])
def predict():
    sync_from_github()
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

    old_content, old_sha = get_file_from_github(EXAMPLES_FILE)
    exemplos_atuais = []
    if old_content:
        try:
            exemplos_atuais = json.loads(old_content)
        except Exception:
            exemplos_atuais = []
    exemplos_finais = exemplos_atuais + data

    status, resp = save_file_to_github(
        EXAMPLES_FILE,
        json.dumps(exemplos_finais, ensure_ascii=False, indent=2),
        "Incrementa exemplos UX",
        sha=old_sha
    )
    if status not in (200, 201):
        return jsonify({"ok": False, "msg": "Erro ao salvar exemplos no GitHub", "resp": resp}), 500

    # Treina e exporta o modelo!
    try:
        # Salva temporário para treino local
        with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(exemplos_finais, f, ensure_ascii=False, indent=2)
        from train_ux import train_and_save_model
        train_and_save_model(EXAMPLES_FILE)
        # Sobe arquivos do modelo pro GitHub
        for fname in [LABELS_FILE, VOCAB_FILE, MODEL_FILE]:
            with open(fname, "rb" if fname == MODEL_FILE else "r", encoding=None if fname == MODEL_FILE else "utf-8") as f:
                content = f.read()
            if fname == MODEL_FILE:
                import base64
                content = base64.b64encode(content).decode("utf-8")
            _, _ = save_file_to_github(fname, content, f"Atualiza {fname}")
        return jsonify({"ok": True, "msg": f"Incrementados {len(data)} exemplos. Modelo treinado e exportado!"})
    except Exception as e:
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
    resp = make_response(str(e) + "\n" + tb, 500)
    resp.headers["Access-Control-Allow-Origin"] = "https://www.ton.com.br"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
