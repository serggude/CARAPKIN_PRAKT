import time
from threading import Thread, Lock
def sequential_chain(n):
    result = 1
    for i in range(n):
        result = (result * 2 + 1) % 1000000
    return result

parallel_result = 0
lock = Lock()

def chain_chunk(start, end):
    global parallel_result

    local_result = 1
    for i in range(start, end):
        local_result = (local_result * 2 + 1) % 1000000

    # попытка сложить результаты — но это математически неверно
    with lock:
        parallel_result += local_result


def parallel_chain(n, num_threads):
    global parallel_result
    parallel_result = 0

    chunk = n // num_threads
    threads = []

    for t in range(num_threads):
        start = t * chunk
        end = (t + 1) * chunk if t != num_threads - 1 else n
        th = Thread(target=chain_chunk, args=(start, end))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    return parallel_result


def measure(func, *args):
    start = time.perf_counter()
    res = func(*args)
    elapsed = (time.perf_counter() - start) * 1000
    return res, elapsed

N = 2_000_000

# последовательная версия
seq_val, seq_time = measure(sequential_chain, N)
print(f"Последовательная версия: {seq_time:.2f} ms")

# 2 потока
par2_val, par2_time = measure(parallel_chain, N, 2)
speed2 = seq_time / par2_time
print(f"2 потока: {par2_time:.2f} ms | ускорение: {speed2:.3f}x")

# 4 потока
par4_val, par4_time = measure(parallel_chain, N, 4)
speed4 = seq_time / par4_time
print(f"4 потока: {par4_time:.2f} ms | ускорение: {speed4:.3f}x")