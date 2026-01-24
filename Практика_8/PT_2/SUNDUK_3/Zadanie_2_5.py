import numpy as np
import networkx as nx
import numba


def laplacian_matrix(G: nx.Graph) -> np.ndarray:
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    L = np.zeros((n, n), dtype=np.float64)

    for v in nodes:
        L[idx[v], idx[v]] = G.degree(v)

    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        L[iu, iv] -= 1.0
        L[iv, iu] -= 1.0

    return L


def find_max_off_diag(A: np.ndarray):
    n = A.shape[0]
    max_val = 0.0
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(A[i, j])
            if val > max_val:
                max_val, p, q = val, i, j
    return p, q, max_val


def compute_rotation_angle(App, Aqq, Apq):
    if App == Aqq:
        theta = np.pi / 4
        c = np.cos(theta)
        s = np.sin(theta)
    else:
        tau = (App - Aqq) / (2.0 * Apq)
        t = 1.0 / (abs(tau) + np.sqrt(1.0 + tau * tau))
        if tau < 0:
            t = -t
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c
    return c, s


@numba.jit(nopython=True, parallel=True)
def jacobi_rotate_parallel(A, V, p, q, c, s):
    n = A.shape[0]

    for i in numba.prange(n):
        if i != p and i != q:
            A_ip = A[i, p]
            A_iq = A[i, q]
            A[i, p] = c * A_ip - s * A_iq
            A[i, q] = s * A_ip + c * A_iq
            A[p, i] = A[i, p]
            A[q, i] = A[i, q]

    App = A[p, p]
    Aqq = A[q, q]
    Apq = A[p, q]
    A[p, p] = c * c * App - 2.0 * s * c * Apq + s * s * Aqq
    A[q, q] = s * s * App + 2.0 * s * c * Apq + c * c * Aqq
    A[p, q] = 0.0
    A[q, p] = 0.0

    for i in numba.prange(n):
        V_ip = V[i, p]
        V_iq = V[i, q]
        V[i, p] = c * V_ip - s * V_iq
        V[i, q] = s * V_ip + c * V_iq


def jacobi_eigenvalues_parallel(A0: np.ndarray, max_iter=2000, tol=1e-10):
    A = A0.copy().astype(np.float64)
    n = A.shape[0]
    V = np.eye(n, dtype=np.float64)

    for _ in range(max_iter):
        p, q, max_val = find_max_off_diag(A)
        if max_val < tol:
            break
        c, s = compute_rotation_angle(A[p, p], A[q, q], A[p, q])
        jacobi_rotate_parallel(A, V, p, q, c, s)

    return np.diag(A).copy()


def lambda2(vals: np.ndarray, tol=1e-12) -> float:
    vals = np.sort(vals)
    for x in vals:
        if x > tol:
            return float(x)
    return float("nan")


def make_connected_graph(n=30, p=0.12, seed=7):
    rng = np.random.default_rng(seed)
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1_000_000)))
        if nx.is_connected(G):
            return G


def main():
    G = make_connected_graph(n=30, p=0.12, seed=7)
    L = laplacian_matrix(G)

    eig_j = jacobi_eigenvalues_parallel(L, max_iter=2000, tol=1e-10)
    eig_e = np.linalg.eigvalsh(L)

    lam2_j = lambda2(eig_j)
    lam2_e = lambda2(eig_e)

    print("lambda2 jacobi :", lam2_j)
    print("lambda2 exact  :", lam2_e)
    print("abs error      :", abs(lam2_j - lam2_e))


if __name__ == "__main__":
    main()
