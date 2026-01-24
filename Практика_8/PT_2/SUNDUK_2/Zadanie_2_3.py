import math
import random
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import Delaunay


def generate_planar_graph(n=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    pos = {i: (random.random(), random.random()) for i in range(n)}
    pts = np.array([pos[i] for i in range(n)])
    tri = Delaunay(pts)

    edges = set()
    for a, b, c in tri.simplices:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        w = math.hypot(x1 - x2, y1 - y2)
        G.add_edge(u, v, weight=float(w))

    # иногда бывает на краю, но обычно Делоне связен
    if not nx.is_connected(G):
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(comp).copy()
        pos = {k: pos[k] for k in G.nodes()}

    return G, pos


def _prim_update_chunk(G, u, chunk, visited, best_w, parent):
    for v in chunk:
        if visited[v]:
            continue
        if G.has_edge(u, v):
            w = G[u][v]["weight"]
            if w < best_w[v]:
                best_w[v] = w
                parent[v] = u


def prim_parallel(G, start=0, num_threads=4):
    nodes = list(G.nodes())
    n = len(nodes)

    visited = {v: False for v in nodes}
    best_w = {v: float("inf") for v in nodes}
    parent = {v: None for v in nodes}

    best_w[start] = 0.0

    mst_edges = []
    total = 0.0

    chunk_size = max(1, (n + num_threads - 1) // num_threads)
    chunks = [nodes[i:i + chunk_size] for i in range(0, n, chunk_size)]

    for _ in range(n):
        u = None
        u_w = float("inf")
        for v in nodes:
            if (not visited[v]) and best_w[v] < u_w:
                u_w = best_w[v]
                u = v

        if u is None:
            break

        visited[u] = True
        total += u_w

        if parent[u] is not None:
            mst_edges.append((parent[u], u, u_w))

        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(_prim_update_chunk, G, u, ch, visited, best_w, parent) for ch in chunks]
            for f in futures:
                f.result()

    return mst_edges, total


def build_mst_graph(nodes, mst_edges):
    T = nx.Graph()
    T.add_nodes_from(nodes)
    for u, v, w in mst_edges:
        T.add_edge(u, v, weight=float(w))
    return T


def pick_A_B_not_direct_in_mst(T, seed=7):
    rng = random.Random(seed)
    nodes = list(T.nodes())
    while True:
        a, b = rng.sample(nodes, 2)
        if not T.has_edge(a, b):
            return a, b


# параллельный Флойд–Уоршелл + восстановление пути
def graph_to_dist_and_next(G):
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    dist = np.full((n, n), np.inf, dtype=np.float64)
    nxt = np.full((n, n), -1, dtype=np.int32)

    for i in range(n):
        dist[i, i] = 0.0
        nxt[i, i] = i

    for u, v, data in G.edges(data=True):
        w = float(data["weight"])
        iu, iv = idx[u], idx[v]
        dist[iu, iv] = w
        dist[iv, iu] = w
        nxt[iu, iv] = iv
        nxt[iv, iu] = iu

    return dist, nxt, nodes, idx


def fw_parallel_phase(dist, nxt, k, start_i, end_i):
    row_k = dist[k, :].copy()

    for i in range(start_i, end_i):
        dik = dist[i, k]
        if np.isinf(dik):
            continue

        cand = dik + row_k
        better = cand < dist[i, :]

        if np.any(better):
            dist[i, better] = cand[better]
            # если i -> k -> j лучше, то первый шаг на пути i->j такой же, как первый шаг i->k
            nxt[i, better] = nxt[i, k]


def floyd_warshall_parallel_with_next(dist0, nxt0, num_threads=4):
    dist = dist0.copy()
    nxt = nxt0.copy()

    n = dist.shape[0]
    chunk = max(1, (n + num_threads - 1) // num_threads)

    for k in range(n):
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = []
            for start_i in range(0, n, chunk):
                end_i = min(n, start_i + chunk)
                futures.append(ex.submit(fw_parallel_phase, dist, nxt, k, start_i, end_i))
            for f in as_completed(futures):
                f.result()

    return dist, nxt


def restore_path(nxt, a_i, b_i):
    """Восстановление пути по матрице nxt (список индексов вершин)."""
    if nxt[a_i, b_i] == -1:
        return None

    path = [a_i]
    cur = a_i

    # защита от бесконечных циклов
    for _ in range(nxt.shape[0] + 5):
        if cur == b_i:
            return path
        cur = int(nxt[cur, b_i])
        if cur == -1:
            return None
        path.append(cur)

    return None


def path_weight(G, path):
    s = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        s += float(G[u][v]["weight"])
    return s


def main():
    G, _ = generate_planar_graph(n=50, seed=42)

    start = list(G.nodes())[0]
    mst_edges, mst_total = prim_parallel(G, start=start, num_threads=4)
    T = build_mst_graph(list(G.nodes()), mst_edges)

    A, B = pick_A_B_not_direct_in_mst(T, seed=7)

    dist0, nxt0, nodes, idx = graph_to_dist_and_next(G)
    dist, nxt = floyd_warshall_parallel_with_next(dist0, nxt0, num_threads=4)

    a_i, b_i = idx[A], idx[B]
    full_len = float(dist[a_i, b_i])

    full_path_idx = restore_path(nxt, a_i, b_i)
    full_path = [nodes[i] for i in full_path_idx] if full_path_idx is not None else None

    mst_path = nx.shortest_path(T, source=A, target=B)
    mst_len = path_weight(T, mst_path)

    print(f"A={A}, B={B} (direct edge in MST: {T.has_edge(A, B)})")
    print(f"Shortest distance in full graph: {full_len:.6f}")
    print(f"Path in full graph: {full_path}")
    print(f"Distance along MST path: {mst_len:.6f}")
    print(f"Path in MST: {mst_path}")
    print()

    print("MST edges:")
    for u, v, w in mst_edges:
        print(f"{u} -- {v} (w={w:.6f})")
    print(f"\nTotal MST weight: {mst_total:.6f}")


if __name__ == "__main__":
    main()
