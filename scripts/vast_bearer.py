#!/usr/bin/env python3
"""Minimal vast.ai REST client (bearer token from $VAST_API_KEY or ~/.vast_api_key).
Money-safety first: `destroy` and `balance` always work; nothing auto-renews.

Commands:
  balance
  search  --gpu "A100 SXM4" --disk 80 [-n 5]
  create  --offer ID --image IMG --disk 80 [--onstart FILE] [--label L]
  info    --id INSTANCE_ID
  list
  destroy --id INSTANCE_ID
"""
import os, sys, json, argparse, urllib.request, urllib.error, urllib.parse
BASE = "https://console.vast.ai/api/v0"

def key():
    k = os.environ.get("VAST_API_KEY")
    p = os.path.expanduser("~/.vast_api_key")
    if not k and os.path.exists(p): k = open(p).read().strip()
    if not k: sys.exit("no vast key ($VAST_API_KEY or ~/.vast_api_key)")
    return k

def req(method, path, body=None):
    url = path if path.startswith("http") else BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", "Bearer " + key())
    if body is not None: r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r) as resp: return json.loads(resp.read() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {method} {url}\n{e.read().decode(errors='replace')[:800]}")

def balance(a):
    d = req("GET", "/users/current/")
    print(f"credit=${d.get('credit')}  balance=${d.get('balance')}  email={d.get('email')}")

def search(a):
    q = {"gpu_name": {"eq": a.gpu}, "rentable": {"eq": True}, "num_gpus": {"eq": 1},
         "type": "on-demand", "disk_space": {"gte": a.disk}, "order": [["dph_total", "asc"]]}
    d = req("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
    for o in d.get("offers", [])[:a.n]:
        print(f"{o['id']}  ${o['dph_total']:.3f}/hr  {o.get('gpu_name')}  "
              f"{o.get('gpu_ram',0)/1024:.0f}GB  disk={o.get('disk_space',0):.0f}GB  "
              f"reliab={o.get('reliability2',0):.2f}  {o.get('geolocation')}  cuda={o.get('cuda_max_good')}")

def create(a):
    body = {"image": a.image, "disk": a.disk, "runtype": "ssh"}
    if a.label: body["label"] = a.label
    if a.onstart: body["onstart"] = open(a.onstart).read()
    d = req("PUT", f"/asks/{a.offer}/", body)
    print(json.dumps(d, indent=2))
    if d.get("success"): print("INSTANCE_ID", d.get("new_contract"))

def info(a):
    d = req("GET", f"https://console.vast.ai/api/v1/instances/{a.id}/")
    i = d.get("instances", d)
    if isinstance(i, list): i = i[0] if i else {}
    print(f"status={i.get('actual_status')} cur_state={i.get('cur_state')} "
          f"ssh={i.get('ssh_host')}:{i.get('ssh_port')} "
          f"dph=${i.get('dph_total')}/hr cost_so_far=${i.get('cost',{}) if isinstance(i.get('cost'),dict) else i.get('inet_up_cost')}")
    print(json.dumps({k: i.get(k) for k in ("id","actual_status","cur_state","ssh_host","ssh_port","dph_total","gpu_name","image_uuid")}, indent=1))

def listi(a):
    d = req("GET", "https://console.vast.ai/api/v1/instances/")
    for i in d.get("instances", []):
        print(f"id={i.get('id')} status={i.get('actual_status')} {i.get('gpu_name')} ${i.get('dph_total')}/hr ssh={i.get('ssh_host')}:{i.get('ssh_port')}")
    if not d.get("instances"): print("(no instances)")

def destroy(a):
    d = req("DELETE", f"/instances/{a.id}/")
    print("destroy ->", json.dumps(d))

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(fn=balance)
    s = sub.add_parser("search"); s.add_argument("--gpu", default="A100 SXM4"); s.add_argument("--disk", type=int, default=80); s.add_argument("-n", type=int, default=5); s.set_defaults(fn=search)
    c = sub.add_parser("create"); c.add_argument("--offer", required=True); c.add_argument("--image", required=True); c.add_argument("--disk", type=int, default=80); c.add_argument("--onstart", default=""); c.add_argument("--label", default=""); c.set_defaults(fn=create)
    fo = sub.add_parser("info"); fo.add_argument("--id", required=True); fo.set_defaults(fn=info)
    sub.add_parser("list").set_defaults(fn=listi)
    de = sub.add_parser("destroy"); de.add_argument("--id", required=True); de.set_defaults(fn=destroy)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
