"""Tiny local web server that shows the brain graph and live-reloads when the
brain changes — so you can watch it evolve while you talk to it (`brain add`,
`brain query`, ...) from another terminal. Stdlib only.
"""
import http.server
import threading
import webbrowser

from brain import db, decay, visualize


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


_POLL_JS = """
<script>
const _BRAIN_V = "%(fp)s";
setInterval(async () => {
  try {
    const v = (await (await fetch('/version')).text()).trim();
    if (v && v !== _BRAIN_V) location.reload();
  } catch (e) {}
}, %(ms)d);
</script>
"""


def render_page(conn, interval: float) -> str:
    """Build the graph HTML and inject the change-polling script."""
    decay.run_decay(conn)
    html = visualize.build_html(conn)
    script = _POLL_JS % {"fp": fingerprint(conn), "ms": int(interval * 1000)}
    return html.replace("</body>", script + "</body>", 1) if "</body>" in html else html + script


def make_handler(interval: float):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: str, ctype="text/html"):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
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
