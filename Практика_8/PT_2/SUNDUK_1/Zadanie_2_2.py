import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def generate_planar_graph(n: int = 50, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    # точки
    pos = {i: (random.random(), random.random()) for i in range(n)}

    try:
        from scipy.spatial import Delaunay
        pts = np.array([pos[i] for i in range(n)])
        tri = Delaunay(pts)

        edges = set()
        for simplex in tri.simplices:
            a, b, c = simplex
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((a, c))))

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            w = math.hypot(x1 - x2, y1 - y2)
            G.add_edge(u, v, weight=w)

        return G, pos

    except Exception:
        G = nx.random_geometric_graph(n, radius=0.25, seed=seed)
        pos = nx.get_node_attributes(G, "pos")

        while not nx.is_connected(G):
            G = nx.random_geometric_graph(n, radius=0.30, seed=seed)
            pos = nx.get_node_attributes(G, "pos")

        # веса по расстоянию
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)

        return G, pos

def _update_chunk(G, u, vertices_chunk, visited, best_w, parent):
    updates = []
    for v in vertices_chunk:
        if visited[v]:
            continue
        if G.has_edge(u, v):
            w = G[u][v]["weight"]
            if w < best_w[v]:
                best_w[v] = w
                parent[v] = u
                updates.append((w, v))
    return updates


def prim_parallel(G: nx.Graph, start: int = 0, num_threads: int = 4):
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    visited = {v: False for v in nodes}
    best_w = {v: float("inf") for v in nodes}
    parent = {v: None for v in nodes}

    best_w[start] = 0.0
    mst_edges = []
    total_cost = 0.0

    for _ in range(n):
        u = None
        u_w = float("inf")
        for v in nodes:
            if not visited[v] and best_w[v] < u_w:
                u_w = best_w[v]
                u = v

        if u is None:
            break  # на всякий случай

        visited[u] = True
        total_cost += u_w

        if parent[u] is not None:
            mst_edges.append((parent[u], u, u_w))

        chunk_size = max(1, n // num_threads)
        chunks = [nodes[i:i + chunk_size] for i in range(0, n, chunk_size)]

        all_updates = []
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(_update_chunk, G, u, ch, visited, best_w, parent) for ch in chunks]
            for f in futures:
                all_updates.extend(f.result())

    return mst_edges, total_cost


def draw_graph_with_mst(G: nx.Graph, pos: dict, mst_edges, save_path="mst_visual.png"):
    plt.figure(figsize=(10, 8))

    # 1) исходный граф — серые тонкие линии
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.8, alpha=0.6)

    # 2) вершины — синие точки
    nx.draw_networkx_nodes(G, pos, node_color="royalblue", node_size=35)

    # 3) MST — красные толстые линии
    mst_edge_list = [(u, v) for (u, v, w) in mst_edges]
    nx.draw_networkx_edges(G, pos, edgelist=mst_edge_list, edge_color="red", width=2.8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.show()


def main():
    G, pos = generate_planar_graph(n=50, seed=42)

    if not nx.is_connected(G):
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()
        pos = {k: pos[k] for k in G.nodes()}

    mst_edges, total_cost = prim_parallel(G, start=0, num_threads=4)

    print("MST edges (u -> v, weight):")
    for u, v, w in mst_edges:
        print(f"{u} -- {v}  (w={w:.4f})")

    print(f"\nTotal MST weight: {total_cost:.4f}")
    draw_graph_with_mst(G, pos, mst_edges, save_path="mst_visual.png")
    print("\nSaved image: mst_visual.png")


if __name__ == "__main__":
    main()
