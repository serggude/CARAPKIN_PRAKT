import time
import random
from threading import Thread, Lock
import matplotlib.pyplot as plt

def sequential_sum(arr):
    return sum(arr)

total_sum = 0
lock = Lock()

def parallel_sum_chunk(arr, start, end):
    global total_sum
    local_sum = sum(arr[start:end])
    with lock:
        total_sum += local_sum

def parallel_sum(arr, num_threads):
    global total_sum
    total_sum = 0

    chunk_size = len(arr) // num_threads
    threads = []

    for i in range(num_threads):
        start = i * chunk_size
        end = start + chunk_size if i != num_threads - 1 else len(arr)
        t = Thread(target=parallel_sum_chunk, args=(arr, start, end))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return total_sum


sizes = [1000, 100_000, 10_000_000]
threads_to_test = [1, 2, 4, 8]

results = {}


for size in sizes:
    print(f"\n=== Размер массива: {size:,} элементов ===")

    array = [random.randint(1, 10) for _ in range(size)]

    # последовательная версия
    start = time.perf_counter()
    seq_res = sequential_sum(array)
    time_seq = (time.perf_counter() - start) * 1000

    print(f"Последовательная версия: {time_seq:.2f} ms")

    results[size] = {
        "seq_time_ms": time_seq,
        "parallel": {}
    }

    # параллельные версии
    for n in threads_to_test:
        start = time.perf_counter()
        par_res = parallel_sum(array, n)
        time_par = (time.perf_counter() - start) * 1000

        if par_res != seq_res:
            print(f"⚠ Ошибка: суммы не совпадают для {n} потоков!")

        speedup = time_seq / time_par if time_par > 0 else 0
        efficiency = (speedup / n) * 100

        print(f"Потоки {n}: {time_par:.2f} ms | "
              f"Ускорение: {speedup:.2f}x | "
              f"Эффективность: {efficiency:.2f}%")

        results[size]["parallel"][n] = {
            "time_ms": time_par,
            "speedup": speedup,
            "efficiency": efficiency
        }

def plot_results_grid(results, threads_to_test):
    for size, data in results.items():
        seq_time = data["seq_time_ms"]

        times = []
        speedups = []
        efficiencies = []

        for n in threads_to_test:
            par = data["parallel"][n]
            times.append(par["time_ms"])
            speedups.append(par["speedup"])
            efficiencies.append(par["efficiency"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Результаты для массива {size:,} элементов", fontsize=16)

        # Время выполнения
        axes[0].plot(threads_to_test, times, marker="o")
        axes[0].set_title("Время (мс)")
        axes[0].set_xlabel("Число потоков")
        axes[0].set_ylabel("Время (мс)")
        axes[0].grid(True)

        # Ускорение
        axes[1].plot(threads_to_test, speedups, marker="o", color="green")
        axes[1].set_title("Ускорение")
        axes[1].set_xlabel("Число потоков")
        axes[1].set_ylabel("Speedup (X)")
        axes[1].grid(True)

        # Эффективность
        axes[2].plot(threads_to_test, efficiencies, marker="o", color="red")
        axes[2].set_title("Эффективность (%)")
        axes[2].set_xlabel("Число потоков")
        axes[2].set_ylabel("Efficiency (%)")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()


plot_results_grid(results, threads_to_test)
