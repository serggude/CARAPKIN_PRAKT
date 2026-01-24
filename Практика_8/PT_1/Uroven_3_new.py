import time
import tracemalloc
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_subset(n_samples=1500, d=120, seed=1):
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)

    X = X[idx].astype(np.float64)
    X -= X.mean(axis=0)

    # уменьшаем размерность: оставляем первые d признаков
    # (можно заменить на random projection, но так проще и быстрее)
    return X[:, :d]


def covariance_matrix(X):
    return (X.T @ X) / (X.shape[0] - 1)


def jacobi_eigenvalues(A0, tol=1e-7, max_iter=4000):
    A = A0.copy().astype(np.float64)
    n = A.shape[0]

    for it in range(max_iter):
        max_val = 0.0
        p, q = 0, 1

        # поиск максимального внедиагонального элемента
        for i in range(n):
            for j in range(i + 1, n):
                v = abs(A[i, j])
                if v > max_val:
                    max_val = v
                    p, q = i, j

        if max_val < tol:
            return np.diag(A).copy(), it

        app, aqq, apq = A[p, p], A[q, q], A[p, q]
        theta = 0.5 * np.arctan2(2.0 * apq, (aqq - app))
        c = np.cos(theta)
        s = np.sin(theta)

        for i in range(n):
            if i == p or i == q:
                continue
            aip, aiq = A[i, p], A[i, q]
            A[i, p] = c * aip - s * aiq
            A[p, i] = A[i, p]
            A[i, q] = s * aip + c * aiq
            A[q, i] = A[i, q]

        A[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0

    return np.diag(A).copy(), max_iter


def benchmark(func, *args, repeats=3, warmup=1, **kwargs):
    # прогрев
    for _ in range(warmup):
        func(*args, **kwargs)

    best_time = 1e100
    best_mem = None
    best_res = None

    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        if dt < best_time:
            best_time = dt
            best_mem = peak
            best_res = res

    return best_res, best_time, best_mem


def main():
    n_samples = 1500
    d = 120

    X = load_mnist_subset(n_samples=n_samples, d=d, seed=7)
    C = covariance_matrix(X)

    (eig_j, iters), t_j, mem_j = benchmark(jacobi_eigenvalues, C, tol=1e-7, max_iter=4000, repeats=2, warmup=0)
    eig_np, t_np, mem_np = benchmark(lambda M: np.linalg.eigh(M)[0], C, repeats=5, warmup=1)

    eig_j = np.sort(eig_j)[::-1]
    eig_np = np.sort(eig_np)[::-1]

    rel_err = np.linalg.norm(eig_j - eig_np) / np.linalg.norm(eig_np)

    print(f"MNIST subset: samples={n_samples}, features={d}, cov={d}x{d}\n")

    print("Jacobi:")
    print(f"  time: {t_j:.3f} s")
    print(f"  iterations: {iters}")
    print(f"  peak memory: {mem_j / 1024**2:.2f} MB")

    print("NumPy eigh:")
    print(f"  time: {t_np:.3f} s")
    print(f"  peak memory: {mem_np / 1024**2:.2f} MB")

    print(f"\nRelative eigenvalue error: {rel_err:.2e}")


if __name__ == "__main__":
    main()
