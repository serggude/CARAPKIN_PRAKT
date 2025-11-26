import multiprocessing as mp
import time
import random

def worker_shared(arr, start, end, out, idx):
    s = 0
    for i in range(start, end):
        s += arr[i]
    out[idx] = s

def sequential_sum(arr):
    return sum(arr)

def parallel_sum_mp_shared(arr, n_processes):
    length = len(arr)
    chunk = length // n_processes

    out = mp.Array('l', n_processes)

    processes = []
    for i in range(n_processes):
        start = i * chunk
        end = (i + 1) * chunk if i != n_processes - 1 else length

        p = mp.Process(target=worker_shared,
                       args=(arr, start, end, out, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return sum(out)

def measure(func, *args):
    t0 = time.perf_counter()
    res = func(*args)
    t = (time.perf_counter() - t0) * 1000
    return res, t

if __name__ == "__main__":
    SIZE = 2_000_000
    print("Размер массива:", SIZE)

    shared_arr = mp.Array('i', SIZE)
    for i in range(SIZE):
        shared_arr[i] = random.randint(1, 10)

    # 1 процесс (последовательно)
    print("\n=== Последовательная версия ===")
    seq_res, seq_t = measure(sequential_sum, shared_arr)
    print(f"Время: {seq_t:.2f} ms")

    # 2 процесса
    print("\n=== 2 процесса ===")
    mp2_res, mp2_t = measure(parallel_sum_mp_shared, shared_arr, 2)
    print(f"Время: {mp2_t:.2f} ms | ускорение: {seq_t/mp2_t:.2f}x")

    # 4 процесса
    print("\n=== 4 процесса ===")
    mp4_res, mp4_t = measure(parallel_sum_mp_shared, shared_arr, 4)
    print(f"Время: {mp4_t:.2f} ms | ускорение: {seq_t/mp4_t:.2f}x")

    print("\nСравнение сумм:")
    print("Последовательно:", seq_res)
    print("2 процесса:", mp2_res)
    print("4 процесса:", mp4_res)
