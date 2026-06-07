"""Tiny local web server that shows the brain graph and live-reloads when the
brain changes — so you can watch it evolve while you talk to it (`brain add`,
`brain query`, ...) from another terminal. Stdlib only.
"""
import http.server
import json
import threading
import webbrowser

from brain import config, db, decay, extract, visualize


def fingerprint(conn) -> str:
    """Cheap signature of graph state; changes on any add/query/touch/edge change.
    (Pure decay doesn't change it — only weight moves, which isn't included — so
    the page doesn't reload spuriously.)"""
    r = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(access_count), 0) a, "
        "COALESCE(MAX(last_accessed), 0) m FROM nodes"
    ).fetchone()
    e = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return f"{r['n']}-{e}-{r['a']}-{r['m']:.0f}"


_UI = """
<div id="brainbar" style="position:fixed;top:0;left:0;right:0;z-index:9999;display:flex;
  gap:8px;padding:8px;background:#16213e;border-bottom:1px solid #0f3460;font-family:sans-serif">
  <input id="brainin" autocomplete="off"
    placeholder="talk to your brain…  (e.g. I started learning the piano)"
    style="flex:1;padding:8px 10px;border-radius:6px;border:1px solid #444;background:#0f0f1e;color:#eee"/>
  <button id="brainbtn" style="padding:8px 16px;border-radius:6px;border:0;background:#4A90D9;color:#fff;cursor:pointer">Add</button>
  <span id="brainmsg" style="color:#9aa;align-self:center;min-width:90px"></span>
</div>
<script>
const _BRAIN_V = "%(fp)s";
async function brainAdd(){
  const i=document.getElementById('brainin'), m=document.getElementById('brainmsg'), b=document.getElementById('brainbtn');
  const t=i.value.trim(); if(!t){return;}
  m.textContent='thinking…'; i.disabled=b.disabled=true;
  try{
    const r=await fetch('/add',{method:'POST',headers:{'Content-Type':'text/plain'},body:t});
    const j=await r.json();
    if(r.ok){ m.textContent='+ '+j.nodes+' nodes'; i.value=''; setTimeout(()=>location.reload(),500); }
    else { m.textContent='error: '+(j.error||r.status); }
  }catch(e){ m.textContent='error'; }
  finally{ i.disabled=b.disabled=false; i.focus(); }
}
document.getElementById('brainbtn').onclick=brainAdd;
document.getElementById('brainin').addEventListener('keydown',e=>{if(e.key==='Enter')brainAdd();});
setInterval(async()=>{ try{ const v=(await (await fetch('/version')).text()).trim();
  if(v && v!==_BRAIN_V) location.reload(); }catch(e){} }, %(ms)d);
</script>
"""


def render_page(conn, interval: float) -> str:
    """Build the graph HTML and inject the talk-to-your-brain bar + change poller."""
    decay.run_decay(conn)
    html = visualize.build_html(conn)
    ui = _UI % {"fp": fingerprint(conn), "ms": int(interval * 1000)}
    return html.replace("</body>", ui + "</body>", 1) if "</body>" in html else html + ui


def make_handler(interval: float):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: str, ctype="text/html"):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            conn = db.connect()
            try:
                if self.path.startswith("/version"):
                    self._send(fingerprint(conn), "text/plain")
                else:
                    self._send(render_page(conn, interval))
            finally:
                conn.close()

        def do_POST(self):
            if not self.path.startswith("/add"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            text = self.rfile.read(length).decode("utf-8", "replace").strip()
            conn = db.connect()
            try:
                if not text:
                    self._json({"error": "empty"}, 400)
                    return
                node_ids, edge_ids = extract.ingest(conn, text, source="web", user=config.get_user())
                self._json({"nodes": len(node_ids), "edges": len(edge_ids)})
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
            finally:
                conn.close()

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def serve(port: int = 8000, interval: float = 3.0, open_browser: bool = True):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(interval))
    url = f"http://127.0.0.1:{port}"
    print(f"Brain live at {url}  (reloads when the brain changes; Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
