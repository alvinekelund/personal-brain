"""Pyvis interactive graph visualization with optional community-detection coloring."""
import webbrowser
import tempfile
from brain import db

TYPE_COLORS = {
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
        "stabilization": {"iterations": 150}
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
        size = 10 + n["weight"] * 28
        if color_by == "cluster":
            color = cluster_colors.get(n["id"], DEFAULT_COLOR)
        else:
            color = TYPE_COLORS.get(n["type"], DEFAULT_COLOR)

        title = (
            f"<b>{n['name']}</b><br>"
            f"type: {n['type']}<br>"
            f"weight: {n['weight']:.2f} &nbsp; confidence: {n['confidence']:.2f}<br><br>"
            f"{n['content'] or ''}"
        )
        net.add_node(n["id"], label=n["name"], title=title, size=size, color=color)

    for e in edges:
        if e["source_id"] in node_ids and e["target_id"] in node_ids:
            width = 1 + e["weight"] * 3
            net.add_edge(
                e["source_id"],
                e["target_id"],
                title=f"{e['relation']}  (w={e['weight']:.2f}, ×{e['reinforcement_count']})",
                label=e["relation"],
                width=width,
            )

    return net.generate_html()


def show(
    conn,
    min_weight: float = 0.0,
    type_filter: str | None = None,
    color_by: str = "type",
):
    """Render the graph and open it in the default browser."""
    html = build_html(conn, min_weight=min_weight, type_filter=type_filter, color_by=color_by)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    return tmp.name
