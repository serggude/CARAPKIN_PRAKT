import time
import concurrent.futures
import multiprocessing

def task1_original(n, a):
    for i in range(1, n):
        a[i] = a[i - 1] * 2 + i * 3
    return a

def task1_transformed_red_black(n, a):
    old = a.copy()
    new = a.copy()

    # четные
    for i in range(2, n, 2):
        new[i] = old[i - 1] * 2 + i * 3

    # нечетные
    for i in range(1, n, 2):
        new[i] = old[i - 1] * 2 + i * 3

    return new

def process_indices_worker(args):
    indices, old = args
    out = []
    for i in indices:
        out.append((i, old[i - 1] * 2 + i * 3))
    return out

def task1_parallel(n, a, num_workers=None):
    old = a.copy()
    new = a.copy()

    even_indices = list(range(2, n, 2))
    odd_indices = list(range(1, n, 2))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        f_even = ex.submit(process_indices_worker, (even_indices, old))
        f_odd = ex.submit(process_indices_worker, (odd_indices, old))

        for i, v in f_even.result():
            new[i] = v
        for i, v in f_odd.result():
            new[i] = v

    return new


def measure(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    return res, time.perf_counter() - t0


def run_experiment():
    sizes = [10_000, 100_000]  # 1000000 для ProcessPool упирается в оперативку 
    workers = multiprocessing.cpu_count()

    print(f"CPU cores: {workers}")
    print("=" * 80)

    for n in sizes:
        print(f"\nn = {n}")

        a_init = [1] * n

        # 1) original (цепочка)
        res_orig, t_orig = measure(task1_original, n, a_init.copy())

        # 2) transformed (последовательно)
        res_tr, t_tr = measure(task1_transformed_red_black, n, a_init.copy())

        # 3) transformed (параллельно)
        res_par, t_par = measure(task1_parallel, n, a_init.copy(), workers)

        print(f"Оригинал (seq):            {t_orig:.4f} сек")
        print(f"Преобразованный (seq):     {t_tr:.4f} сек")
        print(f"Преобразованный (parallel):{t_par:.4f} сек")
        print(f"Совпадение parallel==transformed: {res_par == res_tr}")
        print(f"Совпадение transformed==original: {res_tr == res_orig}")


if __name__ == "__main__":
    run_experiment()
