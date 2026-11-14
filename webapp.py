"""Status web app for the trained bot, for hosting on Fly.io.

    python webapp.py            # listens on 0.0.0.0:${PORT:-8080}

Serves three things:
  /            status page: every run in runs/, its latest checkpoint, and the
               tail of its metrics
  /battles/    the pre-rendered HTML battles from examples/
  /health      a plain 200 for the platform's health checks

Standard library only, so the Docker build is a few seconds.
"""

import html
import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXAMPLES = ROOT / "examples"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_rows():
    rows = []
    if not RUNS.is_dir():
        return rows
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        bot = run_dir / "bot"
        policy = read_json(bot / "policy.json") or {}
        ckpts = sorted((run_dir / "checkpoints").glob("0*")) if (run_dir / "checkpoints").is_dir() else []
        latest = ckpts[-1].name if ckpts else "none"
        steps = int(latest) if latest.isdigit() else 0
        tail = []
        metrics = run_dir / "metrics.jsonl"
        if metrics.is_file():
            try:
                lines = metrics.read_text().splitlines()
                tail = lines[-5:]
            except Exception:
                pass
        rows.append({
            "name": run_dir.name,
            "steps": steps,
            "checkpoint": latest,
            "deck": policy.get("deck") or policy.get("env", {}).get("deck", "?"),
            "metrics": tail,
        })
    return rows


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>clashai status</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;
        background: #10151c; color: #dbe4ee; }}
 h1 {{ color: #7ec8ff; }} table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border: 1px solid #2a3644; padding: 6px 10px; text-align: left; }}
 th {{ background: #1a2330; }} a {{ color: #7ec8ff; }}
 pre {{ background: #1a2330; padding: 8px; overflow-x: auto; font-size: 12px; }}
 .muted {{ color: #7a8a9c; }}
</style></head><body>
<h1>clashai status</h1>
<p class="muted">A PPO-trained Clash Royale bot (Hog 2.6 first, then any deck).
Trained in the royalesim engine; see the repo README for the pipeline.</p>
<h2>Runs</h2>
<table><tr><th>run</th><th>deck</th><th>decisions</th><th>latest checkpoint</th></tr>
{run_rows}
</table>
<h2>Watch it play</h2>
<ul>{battle_links}</ul>
<h2>Metrics tails</h2>
{metric_blocks}
</body></html>"""


def render():
    rows = []
    for r in run_rows():
        rows.append(f"<tr><td>{html.escape(r['name'])}</td><td>{html.escape(str(r['deck']))}</td>"
                    f"<td>{r['steps']:,}</td><td>{html.escape(r['checkpoint'])}</td></tr>")
    run_rows_html = "\n".join(rows) or "<tr><td colspan=4>no runs yet</td></tr>"

    links = []
    if EXAMPLES.is_dir():
        for f in sorted(EXAMPLES.glob("*.html")):
            links.append(f'<li><a href="/battles/{html.escape(f.name)}">{html.escape(f.name)}</a></li>')
    battle_links = "\n".join(links) or "<li>no recorded battles yet</li>"

    blocks = []
    for r in run_rows():
        if r["metrics"]:
            body = html.escape("\n".join(r["metrics"]))
            blocks.append(f"<h3>{html.escape(r['name'])}</h3><pre>{body}</pre>")
    metric_blocks = "\n".join(blocks) or "<p class=muted>no metrics yet</p>"

    return PAGE.format(run_rows=run_rows_html, battle_links=battle_links, metric_blocks=metric_blocks)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, b"ok", "text/plain")
        if path == "/":
            return self._send(200, render().encode())
        if path.startswith("/battles/"):
            name = pathlib.PurePosixPath(path).name
            f = EXAMPLES / name
            if f.is_file():
                return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, b"no such battle", "text/plain")
        return self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"clashai status on 0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
