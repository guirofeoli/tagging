import requests
import base64
import os

# -------- CONFIGURAÇÃO ---------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")   # Busca do ambiente!
GITHUB_OWNER = "guirofeoli"
GITHUB_REPO = "tagging"
GITHUB_BRANCH = "main"
GITHUB_PATH = "railway/"  # Pasta do repositório onde estão os arquivos
# --------------------------------

GITHUB_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/"

def get_file_from_github(filename, is_binary=False):
    """
    Lê um arquivo da pasta railway/ no repo e retorna (conteudo, sha).
    Se is_binary=True, retorna bytes. Caso contrário, retorna string.
    """
    url = GITHUB_API + GITHUB_PATH + filename
    params = {"ref": GITHUB_BRANCH}
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"])
        sha = data["sha"]
        if not is_binary:
            try:
                content = content.decode("utf-8")
            except Exception:
                pass
        return content, sha
    else:
        # Arquivo não existe
        return None, None

def save_file_to_github(filename, content, commit_msg, sha=None, is_binary=False):
    """
    Salva/cria/atualiza um arquivo na pasta railway/ do repo.
    content pode ser str (para arquivos de texto/JSON) ou bytes (para binários).
    Retorna (status_code, response_json)
    """
    url = GITHUB_API + GITHUB_PATH + filename
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    if is_binary and isinstance(content, bytes):
        encoded_content = base64.b64encode(content).decode("utf-8")
    elif isinstance(content, str):
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    else:
        raise Exception("Content precisa ser str ou bytes.")

    payload = {
        "message": commit_msg,
        "content": encoded_content,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha  # Necessário para atualizar um arquivo existente

    resp = requests.put(url, json=payload, headers=headers)
    return resp.status_code, resp.json()

# --------------------------
# Exemplo de uso:
# conteudo, sha = get_file_from_github("ux_examples.json")
# status, res = save_file_to_github("model.bin", binario, "Commit binário", sha, is_binary=True)
# --------------------------
