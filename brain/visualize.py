"""Pyvis interactive graph visualization."""
import webbrowser
import tempfile
from pathlib import Path
from brain import db

TYPE_COLORS = {
    "concept":  "#4A90D9",
    "skill":    "#27AE60",
    "project":  "#E67E22",
    "person":   "#9B59B6",
    "fact":     "#95A5A6",
    "insight":  "#E74C3C",
    "event":    "#F1C40F",
}
DEFAULT_COLOR = "#BDC3C7"


def build_html(conn, min_weight: float = 0.0, type_filter: str | None = None) -> str:
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
          "springLength": 120
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "edges": {"color": {"opacity": 0.5}},
      "interaction": {"hover": true}
    }
    """)

    nodes = db.all_nodes(conn, min_weight=min_weight)
    if type_filter:
        nodes = [n for n in nodes if n["type"] == type_filter]

    node_ids = {n["id"] for n in nodes}

    for n in nodes:
        size = 10 + n["weight"] * 25
        color = TYPE_COLORS.get(n["type"], DEFAULT_COLOR)
        label = n["name"]
        title = (
            f"<b>{n['name']}</b><br>"
            f"type: {n['type']}<br>"
            f"weight: {n['weight']:.2f}<br>"
            f"confidence: {n['confidence']:.2f}<br><br>"
            f"{n['content'] or ''}"
        )
        net.add_node(n["id"], label=label, title=title, size=size, color=color)

    for e in db.all_edges(conn):
        if e["source_id"] in node_ids and e["target_id"] in node_ids:
            net.add_edge(
                e["source_id"],
                e["target_id"],
                title=e["relation"],
                label=e["relation"],
            )

    return net.generate_html()


def show(conn, min_weight: float = 0.0, type_filter: str | None = None):
    """Render the graph and open it in the default browser."""
    html = build_html(conn, min_weight=min_weight, type_filter=type_filter)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    return tmp.name
