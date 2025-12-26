import time
import concurrent.futures
import multiprocessing

def task2_original(n, x, y, z):
    for i in range(1, n):
        x[i] = y[i - 1] + z[i - 1]
        y[i] = x[i - 1] * 2 + z[i - 1]
        z[i] = x[i] + y[i] + i
    return x, y, z

# ПРЕОБРАЗОВАННЫЙ (последовательный)
def task2_transformed_red_black(n, x, y, z):
    old_x, old_y, old_z = x.copy(), y.copy(), z.copy()
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()

    # четные индексы
    for i in range(2, n, 2):
        xi = old_y[i - 1] + old_z[i - 1]
        yi = old_x[i - 1] * 2 + old_z[i - 1]
        zi = xi + yi + i
        new_x[i], new_y[i], new_z[i] = xi, yi, zi

    # нечетные индексы
    for i in range(1, n, 2):
        xi = old_y[i - 1] + old_z[i - 1]
        yi = old_x[i - 1] * 2 + old_z[i - 1]
        zi = xi + yi + i
        new_x[i], new_y[i], new_z[i] = xi, yi, zi

    return new_x, new_y, new_z

def task2_worker(args):
    indices, old_x, old_y, old_z = args
    out = []
    for i in indices:
        xi = old_y[i - 1] + old_z[i - 1]
        yi = old_x[i - 1] * 2 + old_z[i - 1]
        zi = xi + yi + i
        out.append((i, xi, yi, zi))
    return out

def task2_parallel(n, x, y, z, num_workers=None):
    old_x, old_y, old_z = x.copy(), y.copy(), z.copy()
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()

    even_idx = list(range(2, n, 2))
    odd_idx = list(range(1, n, 2))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        f_even = ex.submit(task2_worker, (even_idx, old_x, old_y, old_z))
        f_odd = ex.submit(task2_worker, (odd_idx, old_x, old_y, old_z))

        for i, xi, yi, zi in f_even.result():
            new_x[i], new_y[i], new_z[i] = xi, yi, zi
        for i, xi, yi, zi in f_odd.result():
            new_x[i], new_y[i], new_z[i] = xi, yi, zi

    return new_x, new_y, new_z

def measure(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    return res, time.perf_counter() - t0


def arrays_equal(triple1, triple2):
    x1, y1, z1 = triple1
    x2, y2, z2 = triple2
    return (x1 == x2) and (y1 == y2) and (z1 == z2)

def run_experiment():
    workers = multiprocessing.cpu_count()
    sizes = [10_000, 100_000]

    print(f"CPU cores: {workers}")
    print("=" * 80)

    for n in sizes:
        print(f"\nn = {n}")

        x0 = [0] * n
        y0 = [0] * n
        z0 = [0] * n
        x0[0], y0[0], z0[0] = 1, 2, 3

        # 1) original (последовательный)
        orig_res, t_orig = measure(task2_original, n, x0.copy(), y0.copy(), z0.copy())

        # 2) transformed (последовательный)
        tr_res, t_tr = measure(task2_transformed_red_black, n, x0.copy(), y0.copy(), z0.copy())

        # 3) transformed (параллельный)
        par_res, t_par = measure(task2_parallel, n, x0.copy(), y0.copy(), z0.copy(), workers)

        print(f"Original (seq):             {t_orig:.4f} сек")
        print(f"Transformed (seq):          {t_tr:.4f} сек")
        print(f"Transformed (parallel):     {t_par:.4f} сек")

        print(f"parallel == transformed:    {arrays_equal(par_res, tr_res)}")
        print(f"transformed == original:    {arrays_equal(tr_res, orig_res)}")

        if t_par > 0:
            print(f"Ускорение (orig/par):       {t_orig / t_par:.2f}x")
        else:
            print("Ускорение (orig/par):       n/a")


if __name__ == "__main__":
    run_experiment()
