import multiprocessing as mp
import numpy as np
import time

def compute_chunk(a, b, c, start, end, out, idx):
    out[idx] = np.sum(a[start:end] * b[start:end] + c[start:end])


def parallel_compute(a, b, c, n_processes=4):
    length = len(a)
    chunk = length // n_processes

    # массив частичных сумм
    results = mp.Array('d', n_processes)
    processes = []

    for i in range(n_processes):
        start = i * chunk
        end = (i+1)*chunk if i != n_processes - 1 else length
        p = mp.Process(target=compute_chunk,
                       args=(a, b, c, start, end, results, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return sum(results)


if __name__ == "__main__":
    N = 10_000_000
    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    # ---------------------------
    # Последовательное выполнение
    # ---------------------------
    t0 = time.perf_counter()
    seq_result = np.sum(a * b + c)
    seq_time = (time.perf_counter() - t0) * 1000

    print(f"Последовательное время: {seq_time:.2f} ms")

    # ---------------------------
    # Параллельное выполнение
    # ---------------------------
    t0 = time.perf_counter()
    par_result = parallel_compute(a, b, c, n_processes=4)
    par_time = (time.perf_counter() - t0) * 1000

    print(f"Параллельное время: {par_time:.2f} ms")
    print(f"Ускорение: {seq_time / par_time:.2f}x")

    # Проверка корректности
    print("Корректность:", np.isclose(seq_result, par_result))

    print (seq_result, par_result)
