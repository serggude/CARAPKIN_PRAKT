import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt


# -------------------------------
# Функция для параллельного запуска
# -------------------------------
def apply_filter(matrix):
    return np.sum(matrix * matrix)


# -------------------------------
# Последовательная версия
# -------------------------------
def sequential_processing(matrices):
    results = []
    for m in matrices:
        results.append(apply_filter(m))
    return results


# -------------------------------
# Параллельная версия
# -------------------------------
def parallel_processing(matrices, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(apply_filter, matrices))
    return results


# -------------------------------
# Эксперимент
# -------------------------------
def run_experiment():
    np.random.seed(42)

    num_matrices = 32
    size = 300

    matrices = [np.random.rand(size, size) for _ in range(num_matrices)]

    times = {}

    # Последовательно
    t0 = time.perf_counter()
    sequential_processing(matrices)
    t1 = time.perf_counter()
    times[1] = t1 - t0

    # Параллельно
    for workers in [2, 4, 8]:
        t0 = time.perf_counter()
        parallel_processing(matrices, workers)
        t1 = time.perf_counter()
        times[workers] = t1 - t0

    return times


# -------------------------------
# Граф ускорения
# -------------------------------
def plot_speedup(times):
    workers = sorted(times.keys())
    base_time = times[1]
    speedup = [base_time / times[w] for w in workers]

    plt.figure()
    plt.plot(workers, speedup, marker='o')
    plt.xlabel("Количество процессов")
    plt.ylabel("Ускорение")
    plt.title("Масштабируемость параллельного алгоритма")
    plt.grid(True)
    plt.show()


# -------------------------------
# 🔴 ОБЯЗАТЕЛЬНО!
# -------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()

    times = run_experiment()

    print("Время выполнения:")
    for w, t in times.items():
        print(f"{w} процессов: {t:.4f} сек")

    plot_speedup(times)
