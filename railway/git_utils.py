import requests
import base64

# -------- CONFIGURAÇÃO ---------
GITHUB_TOKEN = "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"  # Troque pelo seu!
GITHUB_OWNER = "guirofeoli"
GITHUB_REPO = "tagging"
GITHUB_BRANCH = "main"
GITHUB_PATH = "railway/"  # Pasta do repositório onde estão os arquivos
# --------------------------------

GITHUB_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/"

def get_file_from_github(filename):
    """
    Lê um arquivo da pasta railway/ no repo e retorna (conteudo, sha).
    """
    url = GITHUB_API + GITHUB_PATH + filename
    params = {"ref": GITHUB_BRANCH}
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        sha = data["sha"]
        return content, sha
    else:
        # Arquivo não existe
        return None, None

def save_file_to_github(filename, content, commit_msg, sha=None):
    """
    Salva/cria/atualiza um arquivo na pasta railway/ do repo.
    Retorna (status_code, response_json)
    """
    url = GITHUB_API + GITHUB_PATH + filename
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha  # Necessário para atualizar um arquivo existente

    resp = requests.put(url, json=payload, headers=headers)
    return resp.status_code, resp.json()

# --------------------------
# Exemplo de uso:
# conteudo, sha = get_file_from_github("ux_examples.json")
# status, res = save_file_to_github("ux_examples.json", '{"foo": "bar"}', "Teste commit", sha)
# --------------------------
