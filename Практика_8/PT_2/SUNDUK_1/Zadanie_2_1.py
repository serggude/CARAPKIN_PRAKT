import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.sparse.csgraph import shortest_path


def generate_dense_graph(n: int, seed: int = 42, w_min: float = 1.0, w_max: float = 10.0) -> np.ndarray:
    """Плотный взвешенный граф как матрица смежности n x n. Диагональ = 0."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(w_min, w_max, size=(n, n)).astype(np.float64)
    np.fill_diagonal(g, 0.0)
    return g


def fw_parallel_phase(D: np.ndarray, k: int, start_i: int, end_i: int) -> None:
    """Одна фаза k, обрабатываем строки i в диапазоне [start_i, end_i)."""
    row_k = D[k, :].copy()  # снимок строки k для чтения
    for i in range(start_i, end_i):
        candidate = D[i, k] + row_k
        D[i, :] = np.minimum(D[i, :], candidate)


def floyd_warshall_parallel(graph: np.ndarray, num_threads: int = 4) -> np.ndarray:
    """Учебная параллельная версия: k последовательно, строки i параллельно."""
    D = graph.copy()
    n = D.shape[0]

    chunk = max(1, (n + num_threads - 1) // num_threads)  # примерно поровну

    for k in range(n):
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = []
            for start_i in range(0, n, chunk):
                end_i = min(n, start_i + chunk)
                futures.append(ex.submit(fw_parallel_phase, D, k, start_i, end_i))

            for f in as_completed(futures):
                f.result()

    return D


def measure_best_time(func, *args, repeats: int = 3, warmup: int = 1, **kwargs) -> float:
    """Берём минимальное время из нескольких запусков (и делаем прогрев)."""
    for _ in range(warmup):
        func(*args, **kwargs)

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def check_correctness():
    """Небольшая проверка, что обе реализации дают одинаковый результат."""
    G = generate_dense_graph(30, seed=123)
    D1 = floyd_warshall_parallel(G, num_threads=4)
    D2 = shortest_path(G, method="FW", directed=True, unweighted=False,
                       overwrite=False, return_predecessors=False)

    ok = np.allclose(D1, D2, atol=1e-9, rtol=0.0)
    print("Check:", "OK" if ok else "FAILED")
    if not ok:
        print("Max diff:", np.max(np.abs(D1 - D2)))


def main():
    sizes = [100, 200, 400]
    num_threads = 4
    repeats = 3
    warmup = 1

    check_correctness()
    print()

    print(f"{'N':>6} | {'our_parallel_fw(s)':>18} | {'scipy_fw(s)':>12} | {'ratio':>8}")
    print("-" * 56)

    for n in sizes:
        G = generate_dense_graph(n, seed=42)

        t_our = measure_best_time(
            floyd_warshall_parallel,
            G,
            num_threads=num_threads,
            repeats=repeats,
            warmup=warmup
        )

        t_scipy = measure_best_time(
            shortest_path,
            G,
            method="FW",
            directed=True,
            unweighted=False,
            overwrite=False,
            return_predecessors=False,
            repeats=repeats,
            warmup=warmup
        )

        ratio = t_our / t_scipy if t_scipy > 0 else float("inf")
        print(f"{n:6d} | {t_our:18.6f} | {t_scipy:12.6f} | {ratio:8.2f}")


if __name__ == "__main__":
    main()
