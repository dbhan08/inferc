#!/usr/bin/env python3
"""Minimal Kaggle kernels client that uses a KGAT_ bearer token (the new Kaggle
API token format), which the classic `kaggle` CLI (basic-auth username:key) rejects.

Auth: bearer token from $KAGGLE_API_TOKEN or ~/.kaggle/kgat_token.
Username: from $KAGGLE_USERNAME or the /hello endpoint.

Commands:
  push <notebook.ipynb> <slug> [--title T] [--gpu] [--internet] [--public]
  status <slug>
  output <slug> [--dest DIR]
  pull   <slug> [--dest DIR]          # fetch the kernel's current .ipynb
where <slug> is just the kernel name (e.g. eval-sota-codebook); the owner is prepended.

Endpoints (Kaggle public API v1): POST /kernels/push, GET /kernels/status,
GET /kernels/output, GET /kernels/pull. Body schema = KernelPushRequest (camelCase).
"""
import os, sys, json, argparse, base64, urllib.request, urllib.error

BASE = "https://www.kaggle.com/api/v1"

def token():
    t = os.environ.get("KAGGLE_API_TOKEN")
    if not t:
        p = os.path.expanduser("~/.kaggle/kgat_token")
        if os.path.exists(p): t = open(p).read().strip()
    if not t: sys.exit("no bearer token ($KAGGLE_API_TOKEN or ~/.kaggle/kgat_token)")
    return t

def req(method, path, body=None, raw=False):
    url = path if path.startswith("http") else BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", "Bearer " + token())
    if body is not None: r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r) as resp:
            b = resp.read()
            return b if raw else json.loads(b or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {method} {url}\n{e.read().decode(errors='replace')[:800]}")

def whoami():
    return os.environ.get("KAGGLE_USERNAME") or req("GET", "/hello")["userName"]

def push(a):
    owner = whoami(); slug = a.slug
    nb = open(a.notebook).read()                     # .ipynb JSON as a string -> `text`
    body = {
        "slug": f"{owner}/{slug}",
        "newTitle": a.title or slug,
        "text": nb,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": not a.public,
        "enableGpu": a.gpu, "enableTpu": False, "enableInternet": a.internet,
        "datasetDataSources": [], "competitionDataSources": [],
        "kernelDataSources": [], "modelDataSources": [], "categoryIds": [],
    }
    r = req("POST", "/kernels/push", body)
    print(json.dumps(r, indent=2))
    if r.get("error"): sys.exit(1)
    print(f"\nrunning: https://www.kaggle.com/code/{owner}/{slug}")

def status(a):
    owner = whoami()
    r = req("GET", f"/kernels/status?userName={owner}&kernelSlug={a.slug}")
    print(json.dumps(r, indent=2))

def output(a):
    owner = whoami()
    r = req("GET", f"/kernels/output?userName={owner}&kernelSlug={a.slug}")
    os.makedirs(a.dest, exist_ok=True)
    for f in r.get("files", []):
        name = f["fileName"]; dst = os.path.join(a.dest, os.path.basename(name))
        if f.get("url"):
            data = req("GET", f["url"], raw=True); open(dst, "wb").write(data)
        elif f.get("content") is not None:
            open(dst, "wb").write(base64.b64decode(f["content"]))
        print("wrote", dst)
    if r.get("log"):
        open(os.path.join(a.dest, "kernel.log"), "w").write(r["log"]); print("wrote log")

def pull(a):
    owner = whoami()
    r = req("GET", f"/kernels/pull?userName={owner}&kernelSlug={a.slug}")
    os.makedirs(a.dest, exist_ok=True)
    blob = r.get("blob", {}); src = blob.get("source", "")
    dst = os.path.join(a.dest, a.slug + ".ipynb"); open(dst, "w").write(src); print("wrote", dst)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("notebook"); p.add_argument("slug")
    p.add_argument("--title", default=""); p.add_argument("--gpu", action="store_true")
    p.add_argument("--internet", action="store_true"); p.add_argument("--public", action="store_true")
    p.set_defaults(fn=push)
    for name, fn in (("status", status), ("output", output), ("pull", pull)):
        q = sub.add_parser(name); q.add_argument("slug")
        if name in ("output", "pull"): q.add_argument("--dest", default="./kaggle_out")
        q.set_defaults(fn=fn)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
