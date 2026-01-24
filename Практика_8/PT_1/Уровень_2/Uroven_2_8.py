import numpy as np
import time
import multiprocessing as mp
from numba import njit, prange
import matplotlib.pyplot as plt


@njit(parallel=True)
def block_multiply(A, B):
    """Numba-ускоренное умножение блоков"""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def worker_block(args):
    """Функция для процесса"""
    A_block, B = args
    return block_multiply(A_block, B)


def hybrid_matrix_multiply(A, B, n_processes):
    """Гибридный алгоритм"""
    n = A.shape[0]
    block_size = n // n_processes

    blocks = []
    for i in range(n_processes):
        start = i * block_size
        end = n if i == n_processes - 1 else (i + 1) * block_size
        blocks.append((A[start:end], B))

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(worker_block, blocks)

    return np.vstack(results)


def pure_process_multiply(A, B):
    """Чисто процессный вариант"""
    return hybrid_matrix_multiply(A, B, mp.cpu_count())


def pure_thread_multiply(A, B):
    """Обычное numpy-умножение (потоки BLAS)"""
    return A @ B


def benchmark_matrix_multiplication():
    print("\n=== МИССИЯ 2.1: БЕНЧМАРК ===")

    n = 2000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # NumPy (потоки)
    t0 = time.time()
    C1 = pure_thread_multiply(A, B)
    t_numpy = time.time() - t0
    print(f"NumPy (потоки BLAS): {t_numpy:.2f} сек")

    # Гибрид
    t0 = time.time()
    C2 = hybrid_matrix_multiply(A, B, mp.cpu_count())
    t_hybrid = time.time() - t0
    print(f"Гибрид (process + numba): {t_hybrid:.2f} сек")




def generate_symmetric_matrix(n, condition_number):
    """Генерация симметричной матрицы с заданным числом обусловленности"""
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    eigvals = np.linspace(1, condition_number, n)
    A = Q @ np.diag(eigvals) @ Q.T
    return A


def jacobi_method(A, b, tol=1e-6, max_iter=10000):
    """Метод Якоби"""
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x) < tol:
            return k
        x = x_new

    return max_iter


def convergence_experiment():
    print("\n=== МИССИЯ 2.2: СХОДИМОСТЬ ЯКОБИ ===")

    n = 50
    condition_numbers = [10, 50, 100, 500, 1000, 5000]
    iterations = []

    for cond in condition_numbers:
        A = generate_symmetric_matrix(n, cond)
        b = np.random.rand(n)
        iters = jacobi_method(A, b)
        iterations.append(iters)
        print(f"cond = {cond:<6} → итераций: {iters}")

    # График
    plt.figure(figsize=(8, 5))
    plt.plot(condition_numbers, iterations, marker='o')
    plt.xscale('log')
    plt.xlabel("Число обусловленности")
    plt.ylabel("Количество итераций")
    plt.title("Сходимость метода Якоби")
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    mp.freeze_support()  # для Windows
    benchmark_matrix_multiplication()
    convergence_experiment()
