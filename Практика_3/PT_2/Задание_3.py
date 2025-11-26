import threading
import time
import numpy as np
import matplotlib.pyplot as plt

def partial_sum(a, start, end, result, idx):
    s = 0
    for i in range(start, end):
        s += a[i]
    result[idx] = s

def measure_time(n_threads, chunk_size, data):
    n = len(data)
    results = [0] * n_threads
    threads = []

    start_time = time.perf_counter()

    for t in range(n_threads):
        start = t * chunk_size
        end = (t + 1) * chunk_size if t < n_threads - 1 else n

        if start >= n:
            break

        thread = threading.Thread(target=partial_sum,
                                  args=(data, start, end, results, t))
        threads.append(thread)
        thread.start()

    for th in threads:
        th.join()

    total = sum(results)
    return time.perf_counter() - start_time


data = np.ones(1_000_000, dtype=np.int32)

chunk_sizes = [10, 50, 100, 500, 1000, 5000,
               10_000, 20_000, 50_000, 100_000, 200_000, 500_000]

thread_options = [2, 4, 8]

results = {t: [] for t in thread_options}

for t in thread_options:
    for chunk in chunk_sizes:
        t_sec = measure_time(t, chunk, data)
        results[t].append(t_sec)
        print(f"Threads={t}, chunk={chunk}, time={t_sec:.6f}s")


plt.figure(figsize=(12, 6))
for t in thread_options:
    plt.plot(chunk_sizes, results[t], label=f"{t} threads")

plt.xscale("log")
plt.xlabel("Размер блока (chunk size)")
plt.ylabel("Время выполнения (сек)")
plt.title("Зависимость времени выполнения от размера блока")
plt.legend()
plt.grid(True)
plt.show()
