from flask import Flask, request, jsonify
import json
import os

# Se estiver usando huggingface/transformers, certifique-se de importar aqui
# from train_ux import train_and_save_model

app = Flask(__name__)

# (Opcional) Healthcheck
@app.route("/")
def home():
    return "Auto-UX API online", 200

# Endpoint para salvar exemplos e treinar o modelo automaticamente
@app.route("/salvar_examples", methods=["POST"])
def salvar_examples():
    import time
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"ok": False, "msg": "Payload deve ser uma lista de exemplos!"}), 400

    # 1. Salva os exemplos recebidos
    try:
        with open("ux_examples.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Auto-UX] Salvos {len(data)} exemplos.")
    except Exception as e:
        print("[Auto-UX] ERRO ao salvar exemplos:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

    # 2. Treinamento automático
    try:
        print("[Auto-UX] Iniciando treinamento automático após salvar exemplos...")
        t0 = time.time()
        # Importação atrasada para evitar problemas de dependências se não precisar treinar
        from train_ux import train_and_save_model
        train_and_save_model("ux_examples.json")
        t1 = time.time()
        print(f"[Auto-UX] Treinamento concluído em {t1-t0:.1f}s.")
        return jsonify({"ok": True, "msg": f"Salvos {len(data)} exemplos e modelo treinado com sucesso!"}), 200
    except Exception as e:
        print("[Auto-UX] ERRO no treino automático:", e)
        return jsonify({"ok": False, "msg": str(e)}), 500

# Adicione outros endpoints que precisar...

if __name__ == "__main__":
    # Use host 0.0.0.0 para o Railway e porta a partir da variável de ambiente
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
