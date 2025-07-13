import requests
import base64
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "guirofeoli"
GITHUB_REPO = "tagging"
GITHUB_BRANCH = "main"
GITHUB_PATH = "railway/"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/"

def get_file_from_github(filename):
    url = GITHUB_API + GITHUB_PATH + filename
    params = {"ref": GITHUB_BRANCH}
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"])
        sha = data["sha"]
        return content, sha
    else:
        print(f"[git_utils] Falha ao baixar {filename}: {resp.status_code} {resp.text}")
        return None, None

def save_file_to_github(filename, content, commit_msg, sha=None, is_binary=False):
    url = GITHUB_API + GITHUB_PATH + filename
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    if is_binary:
        if isinstance(content, bytes):
            b64_content = base64.b64encode(content).decode("utf-8")
        else:
            b64_content = content  # já está base64
        payload = {
            "message": commit_msg,
            "content": b64_content,
            "branch": GITHUB_BRANCH,
            "encoding": "base64"
        }
    else:
        if isinstance(content, str):
            b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        else:
            b64_content = base64.b64encode(content).decode("utf-8")
        payload = {
            "message": commit_msg,
            "content": b64_content,
            "branch": GITHUB_BRANCH
        }
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, json=payload, headers=headers)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {}
