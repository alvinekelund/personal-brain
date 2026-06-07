"""Live local web app for the brain: shows the graph and lets you add, search,
build context, synthesize, reorganize, and inspect status/tree from the browser.
The graph auto-reloads when the brain changes. Stdlib only.
"""
import http.server
import json
import threading
import webbrowser
from urllib.parse import urlparse, parse_qs

from brain import config, db, decay, extract, graph, llm, visualize


def fingerprint(conn) -> str:
    """Cheap signature of graph state; changes on any add/query/touch/edge change."""
    r = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(access_count), 0) a, "
        "COALESCE(MAX(last_accessed), 0) m FROM nodes"
    ).fetchone()
    e = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    return f"{r['n']}-{e}-{r['a']}-{r['m']:.0f}"


# ── API: each returns a JSON-able value ──────────────────────────────────────

def api_query(conn, q, semantic):
    if not q:
        return []
    if semantic and llm.have_key():
        try:
            scored = graph.semantic_search(conn, llm.embed(q), limit=12)
        except Exception as e:
            return [{"error": str(e)[:160]}]
        for _, r in scored:
            db.touch_node(conn, r["id"])
        conn.commit()
        return [{"name": r["name"], "type": r["type"], "score": round(s, 3),
                 "content": (r["content"] or "")[:140]} for s, r in scored]
    res = graph.query_nodes(conn, q)[:12]
    return [{"name": r["name"], "type": r["type"], "weight": round(r["weight"], 2),
             "content": (r["content"] or "")[:140]} for r in res]


def api_context(conn, topic):
    nodes, fb = graph.collect_context_nodes(conn, topic=topic)
    if not nodes:
        return {"doc": "No relevant knowledge found.", "n": 0, "fallback": fb}
    return {"doc": graph.synthesize_context(nodes, topic=topic), "n": len(nodes), "fallback": fb}


def api_status(conn):
    decay.run_decay(conn)
    return {"stats": db.stats(conn), "fading": decay.at_risk_nodes(conn)}


def api_tree(conn, user=None):
    kids = graph.children_map(conn)
    lines, seen = [], set()

    def rec(nid, depth):
        if nid in seen:
            return
        seen.add(nid)
        n = db.get_node(conn, nid)
        if not n:
            return
        lines.append("  " * depth + f"- {n['name']} [{n['type']}] imp={n['importance']:.2f}")
        for c in sorted(kids.get(nid, []),
                        key=lambda c: -(db.get_node(conn, c) or {"importance": 0})["importance"]):
            rec(c, depth + 1)

    root = db.get_node_by_name(conn, user or config.get_user() or "")
    if root:
        rec(root["id"], 0)
    return "\n".join(lines) or "(empty)"


_UI = """
<style>
 #bx{position:fixed;top:0;left:0;right:0;z-index:9999;background:#16213e;border-bottom:1px solid #0f3460;
   font-family:sans-serif;padding:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 #bx input{padding:7px 9px;border-radius:6px;border:1px solid #444;background:#0f0f1e;color:#eee}
 #bx button{padding:7px 12px;border-radius:6px;border:0;background:#0f3460;color:#fff;cursor:pointer}
 #bx button.p{background:#4A90D9}
 #bxmsg{color:#9aa;margin-left:auto;font-size:13px}
 #panel{position:fixed;top:54px;right:0;width:380px;max-height:82vh;overflow:auto;z-index:9998;
   background:rgba(15,15,30,0.95);color:#ddd;font:13px/1.45 monospace;padding:12px;
   border-left:1px solid #0f3460;display:none;white-space:pre-wrap}
 #panel h4{margin:0 0 8px;color:#4A90D9;font-family:sans-serif}
 #panel .close{float:right;cursor:pointer;color:#888}
</style>
<div id="bx">
  <input id="addin" style="flex:1;min-width:220px" placeholder="talk to your brain… (e.g. I started learning piano)"/>
  <button class="p" onclick="bAdd()">Add</button>
  <input id="qin" placeholder="search…" style="width:150px"/>
  <label style="color:#9aa;font-size:12px"><input type="checkbox" id="qsem"/> semantic</label>
  <button onclick="bQuery()">Search</button>
  <input id="cin" placeholder="context topic…" style="width:150px"/>
  <button onclick="bContext()">Context</button>
  <button onclick="bSynth()">Synthesize</button>
  <button onclick="bReorg()">Reorganize</button>
  <button onclick="bStatus()">Status</button>
  <button onclick="bTree()">Tree</button>
  <span id="bxmsg"></span>
</div>
<div id="panel"><span class="close" onclick="document.getElementById('panel').style.display='none'">✕</span><div id="pc"></div></div>
<script>
const _V="%(fp)s", $=id=>document.getElementById(id);
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function show(t,b){$('pc').innerHTML='<h4>'+t+'</h4>'+b;$('panel').style.display='block';}
function msg(m){$('bxmsg').textContent=m;}
async function bAdd(){const i=$('addin'),t=i.value.trim();if(!t)return;msg('thinking…');
 try{const j=await (await fetch('/add',{method:'POST',body:t})).json();
  msg(j.error?('error: '+j.error):('+ '+(j.nodes||0)+' nodes'));if(!j.error){i.value='';setTimeout(()=>location.reload(),500);}}catch(e){msg('error');}}
async function bQuery(){const q=$('qin').value.trim();if(!q)return;const sem=$('qsem').checked?1:0;
 const r=await (await fetch('/query?q='+encodeURIComponent(q)+'&semantic='+sem)).json();
 show('Search: '+esc(q), r.map(x=>'• ['+x.type+'] '+esc(x.name)+(x.score!=null?'  ('+x.score+')':'')+'\\n   '+esc(x.content)).join('\\n')||'(no results)');}
async function bContext(){const t=$('cin').value.trim();if(!t)return;show('Context: '+esc(t),'synthesising…');
 const j=await (await fetch('/context?topic='+encodeURIComponent(t))).json();show('Context: '+esc(t), esc(j.doc));}
async function bSynth(){show('Synthesize','working…');const j=await (await fetch('/synthesize',{method:'POST'})).json();
 show('Synthesize',(j.made||[]).map(m=>'• '+esc(m.source)+' --'+m.relation+'--> '+esc(m.target)).join('\\n')||'(no new links)');setTimeout(()=>location.reload(),700);}
async function bReorg(){show('Reorganize','working…');const j=await (await fetch('/reorganize',{method:'POST'})).json();
 show('Reorganize',(j.edges||0)+' hierarchy edges, '+(j.rescored||0)+' importance updates');setTimeout(()=>location.reload(),700);}
async function bStatus(){const j=await (await fetch('/status')).json(),s=j.stats;
 let h='nodes: '+s.active+' active / '+s.total+' total\\nedges: '+s.edges+'\\navg weight: '+s.avg_weight+'\\nby type: '+esc(JSON.stringify(s.by_type))+'\\n\\nFading soon:\\n';
 h+=(j.fading||[]).map(f=>'• ['+f.type+'] '+esc(f.name)+' w='+f.weight.toFixed(2)).join('\\n')||'(none)';show('Status',h);}
async function bTree(){const t=await (await fetch('/tree')).text();show('Hierarchy',esc(t));}
['addin','qin','cin'].forEach((id,k)=>$(id).addEventListener('keydown',e=>{if(e.key==='Enter')[bAdd,bQuery,bContext][k]();}));
setInterval(async()=>{try{const v=(await (await fetch('/version')).text()).trim();if(v&&v!==_V)location.reload();}catch(e){}},%(ms)d);
</script>
"""


def render_page(conn, interval: float) -> str:
    decay.run_decay(conn)
    html = visualize.build_html(conn)
    ui = _UI % {"fp": fingerprint(conn), "ms": int(interval * 1000)}
    return html.replace("</body>", ui + "</body>", 1) if "</body>" in html else html + ui


def make_handler(interval: float):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, ctype="text/html", status=200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, status=200):
            self._send(json.dumps(obj), "application/json", status)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            conn = db.connect()
            try:
                if u.path == "/version":
                    self._send(fingerprint(conn), "text/plain")
                elif u.path == "/query":
                    self._json(api_query(conn, qs.get("q", [""])[0], qs.get("semantic", ["0"])[0] == "1"))
                elif u.path == "/context":
                    self._json(api_context(conn, qs.get("topic", [""])[0]))
                elif u.path == "/status":
                    self._json(api_status(conn))
                elif u.path == "/tree":
                    self._send(api_tree(conn), "text/plain")
                else:
                    self._send(render_page(conn, interval))
            finally:
                conn.close()

        def do_POST(self):
            path = urlparse(self.path).path
            conn = db.connect()
            try:
                if path == "/add":
                    length = int(self.headers.get("Content-Length", 0) or 0)
                    text = self.rfile.read(length).decode("utf-8", "replace").strip()
                    if not text:
                        self._json({"error": "empty"}, 400)
                        return
                    nids, eids = extract.ingest(conn, text, source="web", user=config.get_user())
                    self._json({"nodes": len(nids), "edges": len(eids)})
                elif path == "/synthesize":
                    self._json({"made": graph.connect_isolated_nodes(conn)})
                elif path == "/reorganize":
                    edges, rescored = extract.reorganize(conn, config.get_user())
                    self._json({"edges": edges, "rescored": rescored})
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
            finally:
                conn.close()

        def log_message(self, *args):
            pass

    return Handler


def serve(port: int = 8000, interval: float = 3.0, open_browser: bool = True):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(interval))
    url = f"http://127.0.0.1:{port}"
    print(f"Brain live at {url}  (add / search / context / synthesize / reorganize in the browser; Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
