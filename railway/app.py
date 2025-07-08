from flask import Flask, request, jsonify, send_file
import json
import os

app = Flask(__name__)

# --- 1. Healthcheck/Root ---
@app.route("/")
def home():
    return "Auto-UX API online", 200

# --- 2. Baixar todos os exemplos salvos (JSON) ---
@app.route("/get_examples", methods=["GET"])
def get_examples():
    try:
        with open("ux_examples.json", "r", encoding="utf-8") as f:
            examples = json.load(f)
        return jsonify({"ok": True, "examples": examples}), 200
    except Exception as e:
        print("[Auto-UX] ERRO ao ler exemplos:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# --- 3. Salvar exemplos e treinar modelo ---
@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    import time
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"ok": False, "msg": "Payload deve ser uma lista de exemplos!"}), 400
    # Salva os exemplos
    try:
        with open("ux_examples.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Auto-UX] Salvos {len(data)} exemplos.")
    except Exception as e:
        print("[Auto-UX] ERRO ao salvar exemplos:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500
    # Treinamento automático
    try:
        print("[Auto-UX] Iniciando treinamento automático após salvar exemplos...")
        t0 = time.time()
        from train_ux import train_and_save_model
        train_and_save_model("ux_examples.json")
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído em {t1-t0:.1f}s.")
        return jsonify({"ok": True, "msg": f"Salvos {len(data)} exemplos e modelo treinado com sucesso!"}), 200
    except Exception as e:
        print("[Auto-UX] ERRO no treino automático:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# --- 4. Predizer sessão a partir de features do front ---
@app.route("/predict", methods=["POST"])
def predict():
    from model_predict import predict_session  # Importe sua função de predição
    features = request.get_json()
    if not features or not isinstance(features, dict):
        return jsonify({"ok": False, "msg": "Payload deve ser um objeto de features!"}), 400
    try:
        label, score = predict_session(features)
        return jsonify({"ok": True, "sessao": label, "score": float(score)})
    except Exception as e:
        print("[Auto-UX] ERRO ao predizer:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# --- 5. Baixar arquivos do modelo treinado (exemplo: model.bin/model.json) ---
@app.route("/download_model", methods=["GET"])
def download_model():
    # Exemplo: baixar um arquivo chamado 'model.bin'
    filename = "model.bin"  # Adapte para seu caso real
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return jsonify({"ok": False, "msg": "Modelo não encontrado!"}), 404

# --- 6. Download dos labels e allWords para usar no front (opcional) ---
@app.route("/get_labels", methods=["GET"])
def get_labels():
    try:
        with open("labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
        return jsonify({"ok": True, "labels": labels})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/get_allwords", methods=["GET"])
def get_allwords():
    try:
        with open("allWords.json", "r", encoding="utf-8") as f:
            words = json.load(f)
        return jsonify({"ok": True, "allWords": words})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

# ----------- FIM DOS ENDPOINTS -----------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
