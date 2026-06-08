"""Pyvis interactive graph visualization with optional community-detection coloring."""
import webbrowser
import tempfile
from brain import db

TYPE_COLORS = {
    "category":     "#F5F5F5",
    "concept":      "#4A90D9",
    "skill":        "#27AE60",
    "project":      "#E67E22",
    "person":       "#9B59B6",
    "organization": "#8E44AD",
    "fact":         "#7F8C8D",
    "insight":      "#E74C3C",
    "event":        "#F1C40F",
    "task":         "#E91E63",
    "artifact":     "#795548",
}
DEFAULT_COLOR = "#BDC3C7"

# Distinct palette for up to 12 communities
CLUSTER_PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#E91E63", "#00BCD4", "#8BC34A",
    "#FF5722", "#607D8B",
]


def _community_colors(nodes, edges) -> dict[str, str]:
    """Run Louvain community detection; return {node_id: hex_color}."""
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        return {}

    G = nx.Graph()
    node_ids = {n["id"] for n in nodes}
    G.add_nodes_from(node_ids)
    for e in edges:
        if e["source_id"] in node_ids and e["target_id"] in node_ids:
            G.add_edge(e["source_id"], e["target_id"], weight=e["weight"])

    if G.number_of_edges() == 0:
        return {}

    communities = louvain_communities(G, seed=42)
    color_map = {}
    for i, community in enumerate(communities):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        for nid in community:
            color_map[nid] = color
    return color_map


def build_html(
    conn,
    min_weight: float = 0.0,
    type_filter: str | None = None,
    color_by: str = "type",   # "type" or "cluster"
) -> str:
    """Return HTML string for the interactive graph."""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis is required: pip install pyvis")

    net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="#eee")
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 130
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 60, "updateInterval": 25},
        "adaptiveTimestep": true
      },
      "edges": {"color": {"opacity": 0.5}, "smooth": {"type": "dynamic"}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    nodes = db.all_nodes(conn, min_weight=min_weight)
    if type_filter:
        nodes = [n for n in nodes if n["type"] == type_filter]

    edges = db.all_edges(conn)
    node_ids = {n["id"] for n in nodes}

    cluster_colors = {}
    if color_by == "cluster":
        cluster_colors = _community_colors(nodes, edges)

    for n in nodes:
        importance = n["importance"] if "importance" in n.keys() else 0.5
        # size by structural role + importance + weight: categories largest, then
        # important nodes; weight adds a little so faded nodes shrink.
        if n["type"] == "category":
            size = 34
        elif n["type"] == "person":
            size = 24 + importance * 12
        else:
            size = 12 + importance * 16 + n["weight"] * 8
        if color_by == "cluster":
            color = cluster_colors.get(n["id"], DEFAULT_COLOR)
        else:
            color = TYPE_COLORS.get(n["type"], DEFAULT_COLOR)

        title = (
            f"<b>{n['name']}</b><br>"
            f"type: {n['type']}<br>"
            f"weight: {n['weight']:.2f} &nbsp; importance: {importance:.2f} "
            f"&nbsp; confidence: {n['confidence']:.2f}<br><br>"
            f"{n['content'] or ''}"
        )
        net.add_node(n["id"], label=n["name"], title=title, size=size, color=color,
                     borderWidth=3 if n["type"] == "category" else 1)

    for e in edges:
        if e["source_id"] not in node_ids or e["target_id"] not in node_ids:
            continue
        if e["relation"] == "part_of":
            # the hierarchy backbone: solid, prominent, arrow toward the parent,
            # no label (the structure speaks for itself)
            net.add_edge(e["source_id"], e["target_id"], color="#6C7A89",
                         width=2.5, arrows="to",
                         title=f"part_of (w={e['weight']:.2f})")
        else:
            # cross-links: lighter and dashed, labelled with the relation
            net.add_edge(e["source_id"], e["target_id"], color="#3a3a5a",
                         width=1 + e["weight"] * 2, dashes=True, label=e["relation"],
                         title=f"{e['relation']}  (w={e['weight']:.2f}, ×{e['reinforcement_count']})")

    return net.generate_html()


_HTML_3D = """<!doctype html><html><head><meta charset="utf-8"><title>Brain (3D)</title>
<style>body{margin:0;background:#0b0b16;overflow:hidden}#g{width:100vw;height:100vh}</style>
<script src="https://unpkg.com/3d-force-graph"></script></head><body>
<div id="g"></div>
<script>
const DATA = __DATA__;
const G = ForceGraph3D()(document.getElementById('g'))
  .backgroundColor('#0b0b16')
  .cooldownTicks(60)
  .graphData(DATA)
  .nodeLabel(n => '<div style="background:#16213e;color:#eee;padding:6px 8px;border-radius:6px;font:13px sans-serif;max-width:280px">'
                 + '<b>' + n.name + '</b> [' + n.type + ']<br>' + (n.desc||'') + '</div>')
  .nodeColor('color').nodeVal('val').nodeOpacity(0.92)
  .linkColor('color').linkWidth(l => l.po ? 1.5 : 0.5).linkOpacity(0.5)
  .linkDirectionalArrowLength(l => l.po ? 3.5 : 0).linkDirectionalArrowRelPos(1)
  .onNodeClick(n => { if (window.brainNodeClick) window.brainNodeClick(n.id); });
</script></body></html>"""


def build_html_3d(conn, min_weight: float = 0.0, type_filter: str | None = None) -> str:
    """Return a standalone 3D force-graph (WebGL via 3d-force-graph, CDN). No
    Python deps — nodes/edges are inlined as JSON."""
    import json
    nodes = db.all_nodes(conn, min_weight=min_weight)
    if type_filter:
        nodes = [n for n in nodes if n["type"] == type_filter]
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in db.all_edges(conn)
             if e["source_id"] in node_ids and e["target_id"] in node_ids]

    def imp(n):
        return n["importance"] if "importance" in n.keys() else 0.5

    gnodes = [{
        "id": n["id"], "name": n["name"], "type": n["type"],
        "color": TYPE_COLORS.get(n["type"], DEFAULT_COLOR),
        "val": 10 if n["type"] == "category" else 2 + imp(n) * 6,
        "desc": (n["content"] or "")[:160],
    } for n in nodes]
    glinks = [{
        "source": e["source_id"], "target": e["target_id"],
        "po": e["relation"] == "part_of",
        "color": "#8a97a6" if e["relation"] == "part_of" else "#3a3a5a",
    } for e in edges]
    return _HTML_3D.replace("__DATA__", json.dumps({"nodes": gnodes, "links": glinks}))


def show(
    conn,
    min_weight: float = 0.0,
    type_filter: str | None = None,
    color_by: str = "type",
    threed: bool = False,
):
    """Render the graph and open it in the default browser (2D Pyvis, or 3D WebGL)."""
    if threed:
        html = build_html_3d(conn, min_weight=min_weight, type_filter=type_filter)
    else:
        html = build_html(conn, min_weight=min_weight, type_filter=type_filter, color_by=color_by)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    return tmp.name
