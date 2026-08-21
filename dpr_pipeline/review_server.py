"""Local, editable version of the AST tree viewer — browse the hierarchy,
click a node, fix its text or a misassigned heading level, and save; the
change is written straight back to the ast.json file on disk. A static
published artifact can't do the "write back to disk" part, so this runs as
a small local server instead (stdlib http.server, no new dependency).

Usage:
    uv run python -m dpr_pipeline.review_server ast.json [--port 8765]

Then open http://localhost:8765 in a browser.
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

TAG_RE = re.compile(r"<[^>]+>")


def node_at_path(root: dict, path: list) -> dict:
    node = root
    for index in path:
        node = node["children"][index]
    return node


def annotate_paths(node: dict, path: list = ()) -> None:
    node["_path"] = list(path)
    for i, child in enumerate(node.get("children", [])):
        annotate_paths(child, path + (i,))


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AST Review — {title}</title>
<style>
:root {{
  --bg: #eef2f4; --surface: #ffffff; --surface-2: #e3eaee; --ink: #1b2a33; --ink-soft: #4c6270;
  --rule: #c7d3d8; --accent: #2f6690; --accent-soft: #dbe8ee; --ok: #1d7a5f; --warn: #b3452f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #10161b; --surface: #182229; --surface-2: #1f2a32; --ink: #dce6ea; --ink-soft: #93a7b1;
    --rule: #2a3840; --accent: #6fb3d8; --accent-soft: #223744; --ok: #5fcaa3; --warn: #e08a76; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Georgia, serif; background: var(--bg); color: var(--ink); display: flex; height: 100vh; }}
.mono {{ font-family: ui-monospace, Consolas, monospace; }}
.tree-pane {{ width: 40%; overflow: auto; border-right: 1px solid var(--rule); background: var(--surface); padding: 14px 8px; }}
.detail-pane {{ flex: 1; overflow: auto; padding: 20px 30px; }}
ul.tree, ul.tree ul {{ list-style: none; margin: 0; padding-left: 20px; }}
ul.tree {{ padding-left: 4px; }}
li {{ position: relative; }}
.row {{ display: flex; gap: 6px; padding: 3px 6px; border-radius: 3px; cursor: pointer; font-size: 13px; }}
.row:hover {{ background: var(--surface-2); }}
.row.selected {{ background: var(--accent-soft); }}
.row.sec .label {{ font-weight: 700; color: var(--accent); }}
.row.dirty::after {{ content: '\\25CF'; color: var(--warn); margin-left: 4px; }}
.label {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.page {{ color: var(--ink-soft); font-size: 11px; }}
textarea {{ width: 100%; min-height: 160px; font-family: ui-monospace, Consolas, monospace; font-size: 13px;
  padding: 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--ink); }}
input[type=number], input[type=text] {{ font-family: ui-monospace, Consolas, monospace; padding: 6px 8px;
  border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink); }}
.field {{ margin-bottom: 16px; }}
.field label {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-soft); margin-bottom: 4px; }}
button {{ background: var(--accent); color: white; border: none; padding: 8px 18px; border-radius: 5px; cursor: pointer; font-size: 13px; }}
button:hover {{ opacity: 0.9; }}
.status {{ margin-left: 12px; font-size: 12px; }}
.status.ok {{ color: var(--ok); }}
.status.err {{ color: var(--warn); }}
.empty {{ color: var(--ink-soft); text-align: center; padding-top: 60px; }}
</style>
</head>
<body>
<div class="tree-pane"><ul class="tree" id="tree-root"></ul></div>
<div class="detail-pane" id="detail-pane"><p class="empty">Click a node to edit it.</p></div>

<script id="tree-data" type="application/json">{tree_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('tree-data').textContent);
let selectedPath = null;

function tagStrip(html) {{
  const d = document.createElement('div'); d.innerHTML = html;
  return (d.textContent || '').replace(/\\s+/g, ' ').trim();
}}

function buildTree(node, container) {{
  const li = document.createElement('li');
  const row = document.createElement('div');
  row.className = 'row' + (node.type === 'section' ? ' sec' : '');
  row.dataset.path = JSON.stringify(node._path);

  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = node.type === 'section' ? `H${{node.level}} ${{node.title}}` : tagStrip(node.html || '').slice(0, 60);
  row.appendChild(label);

  if (node.page) {{
    const pg = document.createElement('span'); pg.className = 'page'; pg.textContent = 'p' + node.page;
    row.appendChild(pg);
  }}
  row.onclick = () => selectNode(node._path);
  li.appendChild(row);

  const children = node.children || [];
  if (children.length) {{
    const ul = document.createElement('ul'); ul.className = 'tree';
    children.forEach(c => buildTree(c, ul));
    li.appendChild(ul);
  }}
  container.appendChild(li);
}}

function findNode(path) {{
  let node = DATA;
  for (const i of path) node = node.children[i];
  return node;
}}

function selectNode(path) {{
  selectedPath = path;
  document.querySelectorAll('.row.selected').forEach(r => r.classList.remove('selected'));
  const row = document.querySelector(`.row[data-path='${{JSON.stringify(path)}}']`);
  if (row) row.classList.add('selected');

  const node = findNode(path);
  const pane = document.getElementById('detail-pane');
  let fields = '';
  if (node.type === 'section') {{
    fields = `
      <div class="field"><label>Title</label><input type="text" id="f-title" value="${{node.title.replace(/"/g,'&quot;')}}"></div>
      <div class="field"><label>Level (h1-h6)</label><input type="number" id="f-level" value="${{node.level}}" min="1" max="6"></div>`;
  }} else {{
    fields = `
      <div class="field"><label>Label</label><input type="text" id="f-label" value="${{node.label}}"></div>
      <div class="field"><label>Content (HTML)</label><textarea id="f-html">${{node.html || ''}}</textarea></div>`;
  }}

  pane.innerHTML = `
    <p class="mono" style="color:var(--ink-soft);font-size:12px;">page ${{node.page || '-'}}</p>
    ${{fields}}
    <button onclick="save()">Save changes</button>
    <span class="status" id="status"></span>`;
}}

async function save() {{
  const node = findNode(selectedPath);
  const patch = {{ path: selectedPath }};
  if (node.type === 'section') {{
    patch.title = document.getElementById('f-title').value;
    patch.level = parseInt(document.getElementById('f-level').value, 10);
  }} else {{
    patch.label = document.getElementById('f-label').value;
    patch.html = document.getElementById('f-html').value;
  }}

  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Saving...';
  statusEl.className = 'status';
  try {{
    const resp = await fetch('/save', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify(patch) }});
    if (!resp.ok) throw new Error(await resp.text());
    Object.assign(node, patch);
    statusEl.textContent = 'Saved to ast.json';
    statusEl.className = 'status ok';
    document.querySelector(`.row[data-path='${{JSON.stringify(selectedPath)}}']`).classList.add('dirty');
    rebuildLabel(selectedPath, node);
  }} catch (e) {{
    statusEl.textContent = 'Save failed: ' + e.message;
    statusEl.className = 'status err';
  }}
}}

function rebuildLabel(path, node) {{
  const row = document.querySelector(`.row[data-path='${{JSON.stringify(path)}}']`);
  const label = row.querySelector('.label');
  label.textContent = node.type === 'section' ? `H${{node.level}} ${{node.title}}` : tagStrip(node.html || '').slice(0, 60);
}}

DATA.children.forEach(c => buildTree(c, document.getElementById('tree-root')));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    tree = None
    tree_path = None

    def log_message(self, format, *args):
        pass  # keep stdout clean; errors still raise normally

    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        html = PAGE_TEMPLATE.format(
            title=self.tree.get("title", ""),
            tree_json=json.dumps(self.tree),
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/save":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        patch = json.loads(self.rfile.read(length))

        try:
            node = node_at_path(self.tree, patch["path"])
            for key in ("title", "level", "label", "html"):
                if key in patch:
                    node[key] = patch[key]

            with open(self.tree_path, "w", encoding="utf-8") as f:
                clean = strip_paths(self.tree)
                json.dump(clean, f, indent=2, ensure_ascii=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))


def strip_paths(node):
    clean = {k: v for k, v in node.items() if k != "_path"}
    if "children" in clean:
        clean["children"] = [strip_paths(c) for c in clean["children"]]
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local editable AST review server")
    parser.add_argument("tree_json")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)
    annotate_paths(tree)

    Handler.tree = tree
    Handler.tree_path = args.tree_json

    server = HTTPServer(("localhost", args.port), Handler)
    print(f"Serving {args.tree_json} at http://localhost:{args.port} — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
