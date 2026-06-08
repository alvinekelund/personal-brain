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


def api_ask(conn, q, history=None):
    return graph.answer_question(conn, q, history=history) if q else {"answer": "", "sources": []}


def api_context(conn, topic):
    nodes, fb = graph.collect_context_nodes(conn, topic=topic)
    if not nodes:
        return {"doc": "No relevant knowledge found.", "n": 0, "fallback": fb}
    return {"doc": graph.synthesize_context(nodes, topic=topic), "n": len(nodes), "fallback": fb}


def api_node(conn, node_id):
    n = db.get_node(conn, node_id)
    if not n:
        return {"error": "not found"}
    edges = []
    for e in db.edges_for_node(conn, node_id):
        other = db.get_node(conn, e["target_id"] if e["source_id"] == node_id else e["source_id"])
        if other:
            edges.append({"rel": e["relation"], "other": other["name"],
                          "dir": "→" if e["source_id"] == node_id else "←"})
    return {"name": n["name"], "type": n["type"], "content": n["content"] or "",
            "importance": round(n["importance"], 2), "weight": round(n["weight"], 2),
            "edges": edges}


def api_status(conn):
    decay.run_decay(conn)
    return {"stats": db.stats(conn), "fading": decay.at_risk_nodes(conn),
            "areas": graph.category_breakdown(conn, config.get_user())}


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
 :root{--accent:#5b8def;--text:#eceefb;--muted:#9aa0b8;--border:#2a2f4f;--card:#1a1d33;}
 #bx{position:fixed;top:0;left:0;right:0;z-index:9999;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
   padding:9px 14px;background:linear-gradient(180deg,#1b1f3a,#141729);border-bottom:1px solid var(--border);
   box-shadow:0 3px 14px rgba(0,0,0,.45);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .brand{font-weight:700;color:var(--text);font-size:15px;white-space:nowrap;letter-spacing:.2px}
 .brand b{color:var(--accent)}
 .inp{padding:8px 11px;border-radius:9px;border:1px solid var(--border);background:#0d0f20;color:var(--text);
   font-size:13px;outline:none;font-family:inherit;transition:border-color .12s}
 .inp:focus{border-color:var(--accent)}
 .btn{padding:8px 13px;border-radius:9px;border:1px solid var(--border);background:#23274a;color:var(--text);
   cursor:pointer;font-size:13px;font-family:inherit;transition:.12s;white-space:nowrap}
 .btn:hover{background:#2d3360;border-color:var(--accent)}
 .btn.primary{background:linear-gradient(180deg,var(--accent),#4470cf);border:0;color:#fff;font-weight:600}
 .btn.primary:hover{filter:brightness(1.12)}
 .sep{width:1px;height:22px;background:var(--border)}
 .seg{display:flex;border:1px solid var(--border);border-radius:9px;overflow:hidden}
 .seg a{padding:7px 12px;color:var(--muted);text-decoration:none;font-size:12px;font-weight:600}
 .seg a.on,.seg a:hover{background:var(--accent);color:#fff}
 .ctl{display:flex;align-items:center;gap:5px;color:var(--muted);font-size:12px}
 #bxmsg{color:var(--muted);font-size:12px;margin-left:auto;white-space:nowrap}
 #panel{position:fixed;top:62px;right:14px;width:410px;max-height:80vh;z-index:9998;background:var(--card);
   border:1px solid var(--border);border-radius:16px;box-shadow:0 16px 48px rgba(0,0,0,.6);display:none;
   overflow:hidden;font-family:system-ui,sans-serif;animation:pin .16s ease}
 @keyframes pin{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:none}}
 #phead{display:flex;align-items:center;justify-content:space-between;padding:13px 16px;
   border-bottom:1px solid var(--border);font-weight:600;color:var(--text);font-size:14px}
 #phead .x{cursor:pointer;color:var(--muted);font-size:20px;line-height:1}
 #phead .x:hover{color:var(--text)}
 #pc{padding:14px 16px;max-height:68vh;overflow:auto;color:#cfd3ea;font-size:13px;line-height:1.6}
 #pc b{color:var(--accent)}#pc i{color:var(--muted)}
</style>
<div id="bx">
  <span class="brand">🧠 <b>brain</b></span>
  <input id="addin" class="inp" style="flex:1;min-width:160px" placeholder="add a thought…"/>
  <button class="btn primary" onclick="bAdd()">Add</button>
  <span class="sep"></span>
  <input id="askin" class="inp" style="width:130px" placeholder="ask…"/>
  <button class="btn primary" onclick="bAsk()">Ask</button>
  <input id="qin" class="inp" style="width:100px" placeholder="search…"/>
  <label class="ctl"><input type="checkbox" id="qsem"/>sem</label>
  <button class="btn" onclick="bQuery()">Search</button>
  <input id="cin" class="inp" style="width:100px" placeholder="context…"/>
  <button class="btn" onclick="bContext()">Context</button>
  <span class="sep"></span>
  <button class="btn" onclick="bDigest()">Digest</button>
  <button class="btn" onclick="bStatus()">Status</button>
  <button class="btn" onclick="bTree()">Tree</button>
  <button class="btn" onclick="bSynth()">Synthesize</button>
  <button class="btn" onclick="bReorg()">Reorganize</button>
  <span class="sep"></span>
  <span class="seg"><a href="?view=2d" id="t2d">2D</a><a href="?view=3d" id="t3d">3D</a></span>
  <label class="ctl">min<input type="range" id="mw" min="0" max="1" step="0.05" style="width:72px" onchange="setMin(this.value)"/></label>
  <span id="bxmsg"></span>
</div>
<div id="panel">
  <div id="phead"><span id="ptitle"></span><span class="x" onclick="document.getElementById('panel').style.display='none'">&times;</span></div>
  <div id="pc"></div>
</div>
<script>
const _V="%(fp)s", $=id=>document.getElementById(id);
const refresh=()=>window.brainRefresh?window.brainRefresh():location.reload();
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function show(t,b){$('ptitle').innerHTML=t;$('pc').innerHTML=b;$('panel').style.display='block';}
function msg(m){$('bxmsg').textContent=m;}
async function bAdd(){const i=$('addin'),t=i.value.trim();if(!t)return;msg('thinking…');
 try{const j=await (await fetch('/add',{method:'POST',body:t})).json();
  msg(j.error?('error: '+j.error):('+ '+(j.nodes||0)+' nodes'));if(!j.error){i.value='';setTimeout(refresh,500);}}catch(e){msg('error');}}
async function bQuery(){const q=$('qin').value.trim();if(!q)return;const sem=$('qsem').checked?1:0;
 const r=await (await fetch('/query?q='+encodeURIComponent(q)+'&semantic='+sem)).json();
 show('Search: '+esc(q), r.map(x=>'• ['+x.type+'] '+esc(x.name)+(x.score!=null?'  ('+x.score+')':'')+'\\n   '+esc(x.content)).join('\\n')||'(no results)');}
async function bContext(){const t=$('cin').value.trim();if(!t)return;show('Context: '+esc(t),'synthesising…');
 const j=await (await fetch('/context?topic='+encodeURIComponent(t))).json();show('Context: '+esc(t), esc(j.doc));}
let chat=[];
function renderChat(){ show('Ask', chat.map(c=>'<b>Q:</b> '+esc(c.q)+'<br><b>A:</b> '+esc(c.a)+(c.src?'<br><span style="color:#789">sources: '+esc(c.src)+'</span>':'')).join('<br><br>')); }
async function bAsk(){const q=$('askin').value.trim();if(!q)return;$('askin').value='';
 chat.push({q:q,a:'thinking…',src:''}); renderChat();
 try{ const j=await (await fetch('/ask',{method:'POST',body:JSON.stringify({q:q,history:chat.slice(0,-1).map(c=>({q:c.q,a:c.a}))})})).json();
   chat[chat.length-1].a=j.answer; chat[chat.length-1].src=(j.sources||[]).join(', '); }
 catch(e){ chat[chat.length-1].a='error'; } renderChat(); }
async function bSynth(){show('Synthesize','working…');const j=await (await fetch('/synthesize',{method:'POST'})).json();
 show('Synthesize',(j.made||[]).map(m=>'• '+esc(m.source)+' --'+m.relation+'--> '+esc(m.target)).join('\\n')||'(no new links)');setTimeout(refresh,700);}
async function bReorg(){show('Reorganize','working…');const j=await (await fetch('/reorganize',{method:'POST'})).json();
 show('Reorganize',(j.edges||0)+' hierarchy edges, '+(j.rescored||0)+' importance updates');setTimeout(refresh,700);}
async function bStatus(){const j=await (await fetch('/status')).json(),s=j.stats;
 let h='nodes: '+s.active+' active / '+s.total+' total\\nedges: '+s.edges+'\\navg weight: '+s.avg_weight+'\\nby type: '+esc(JSON.stringify(s.by_type))+'\\n\\nBy area:\\n';
 h+=(j.areas||[]).map(a=>'• '+esc(a[0])+' ('+a[1]+')').join('\\n')||'(none)';h+='\\n\\nFading soon:\\n';
 h+=(j.fading||[]).map(f=>'• ['+f.type+'] '+esc(f.name)+' w='+f.weight.toFixed(2)).join('\\n')||'(none)';show('Status',h);}
async function bTree(){const t=await (await fetch('/tree')).text();show('Hierarchy',esc(t));}
async function bDigest(){const j=await (await fetch('/digest')).json();
 let h='<b>Top of mind</b><br>'+(j.top||[]).map(t=>'• ['+t.type+'] '+esc(t.name)+' ('+t.importance+')').join('<br>');
 if((j.tasks||[]).length)h+='<br><br><b>Open tasks</b><br>'+j.tasks.map(esc).join('<br>');
 if((j.fading||[]).length)h+='<br><br><b>Fading soon</b><br>'+j.fading.map(f=>'• '+esc(f.name)).join('<br>');
 if((j.areas||[]).length)h+='<br><br><b>By area</b><br>'+j.areas.map(a=>esc(a[0])+' ('+a[1]+')').join('<br>');
 show('Digest',h);}
function showNode(j){ if(j.error){show('Node',j.error);return;}
 let b='importance '+j.importance+' · weight '+j.weight+'<br><br>'+esc(j.content)+'<br><br><b>connections</b><br>';
 b+=(j.edges||[]).map(e=>'• '+e.dir+' '+e.rel+' '+esc(e.other)).join('<br>')||'(none)';
 show(esc(j.name)+' ['+j.type+']', b); }
window.brainNodeClick=async id=>{ if(!id)return; showNode(await (await fetch('/node?id='+encodeURIComponent(id))).json()); };
(function hook2D(){ if(typeof network!=='undefined' && network.on){
   network.on('click',p=>{ if(p.nodes&&p.nodes[0]) window.brainNodeClick(p.nodes[0]); }); }
 else setTimeout(hook2D,300); })();
function setMin(v){const u=new URL(location);u.searchParams.set('min',v);location=u;}
(function(){const u=new URL(location),m=u.searchParams.get('min');if(m)$('mw').value=m;
 const tv=u.searchParams.get('view')==='3d'?'t3d':'t2d';const e=$(tv);if(e)e.classList.add('on');})();
['addin','qin','cin','askin'].forEach((id,k)=>$(id).addEventListener('keydown',e=>{if(e.key==='Enter')[bAdd,bQuery,bContext,bAsk][k]();}));
setInterval(async()=>{try{const v=(await (await fetch('/version')).text()).trim();if(v&&v!==_V)refresh();}catch(e){}},%(ms)d);
</script>
"""


def render_page(conn, interval: float, view: str = "2d", min_weight: float = 0.0) -> str:
    decay.run_decay(conn)
    if view == "3d":
        html = visualize.build_html_3d(conn, min_weight=min_weight)
    else:
        html = visualize.build_html_live(conn, min_weight=min_weight)
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
                elif u.path == "/ask":
                    self._json(api_ask(conn, qs.get("q", [""])[0]))
                elif u.path == "/status":
                    self._json(api_status(conn))
                elif u.path == "/digest":
                    self._json(graph.digest(conn, config.get_user()))
                elif u.path == "/tree":
                    self._send(api_tree(conn), "text/plain")
                elif u.path == "/node":
                    self._json(api_node(conn, qs.get("id", [""])[0]))
                elif u.path == "/graph":
                    try:
                        mw = float(qs.get("min", ["0"])[0])
                    except ValueError:
                        mw = 0.0
                    self._json(visualize.graph_data(conn, min_weight=mw))
                else:
                    try:
                        mw = float(qs.get("min", ["0"])[0])
                    except ValueError:
                        mw = 0.0
                    self._send(render_page(conn, interval, qs.get("view", ["2d"])[0], mw))
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
                elif path == "/ask":
                    length = int(self.headers.get("Content-Length", 0) or 0)
                    body = self.rfile.read(length).decode("utf-8", "replace")
                    try:
                        data = json.loads(body) if body else {}
                    except ValueError:
                        data = {}
                    self._json(api_ask(conn, (data.get("q") or "").strip(), data.get("history") or []))
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
