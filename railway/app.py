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
NUMERIC_MEANS_FILE = "numeric_means.json"

# --- Helper para sincronizar todos arquivos do GitHub para o container ---
def sync_from_github():
    for fname in [EXAMPLES_FILE, LABELS_FILE, VOCAB_FILE, NUMERIC_MEANS_FILE]:
        content, _ = get_file_from_github(fname)
        if content is not None:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(content)

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

@app.route("/sync_from_github", methods=["POST"])
def sync_endpoint():
    sync_from_github()
    return jsonify({"ok": True, "msg": "Sincronizado do GitHub para local."})

@app.route("/get_labels", methods=["GET"])
def get_labels():
    sync_from_github()
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return jsonify({"ok": True if labels else False, "labels": labels})
    except Exception:
        return jsonify({"ok": False, "labels": []})

@app.route("/predict", methods=["POST"])
def predict():
    sync_from_github()  # <- sempre sincroniza antes de predizer!
    try:
        from model_predict import predict_session
        features = request.get_json()
        print("[Auto-UX] Features recebidas:", features)
        label, score = predict_session(features)
        print("[Auto-UX] Predição:", label, score)
        return jsonify({"ok": True, "sessao": label, "score": score})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[Auto-UX] ERRO no predict:", e)
        print(tb)
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

    try:
        import time
        from train_ux import train_and_save_model
        with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(exemplos_finais, f, ensure_ascii=False, indent=2)
        train_and_save_model(EXAMPLES_FILE)
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído.")
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
    app.run(host="0.0.0.0", port=8080)
