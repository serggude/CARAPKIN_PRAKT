import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba as nb
import multiprocessing as mp


# чтобы numpy/BLAS не делали "скрытую" многопоточность (иначе сравнение нечестное)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ---- Numba: блочное умножение C[i0:i1, :] = A[i0:i1, :] @ B ----
@nb.njit(fastmath=True)
def matmul_blocked_rows(A, B, C, i0, i1, BS):
    n = A.shape[0]
    m = B.shape[1]
    kdim = A.shape[1]

    for i in range(i0, i1):
        for jj in range(0, m, BS):
            j_end = min(jj + BS, m)
            for kk in range(0, kdim, BS):
                k_end = min(kk + BS, kdim)

                for j in range(jj, j_end):
                    s = 0.0
                    for k in range(kk, k_end):
                        s += A[i, k] * B[k, j]
                    C[i, j] += s


def make_matrices(n=2000, seed=1, dtype=np.float32):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=dtype)
    B = rng.random((n, n), dtype=dtype)
    return A, B


def chunks(n, rows_per_chunk):
    out = []
    for i0 in range(0, n, rows_per_chunk):
        out.append((i0, min(n, i0 + rows_per_chunk)))
    return out


def time_best(func, *args, repeats=2, warmup=0, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)

    best = 10**18
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


# ------------------------------
# 1) Pure threads (numpy @ внутри)
# ------------------------------
def matmul_threads_numpy(A, B, rows_per_chunk=200, workers=8):
    n = A.shape[0]
    C = np.empty((n, n), dtype=A.dtype)

    def work(i0, i1):
        C[i0:i1, :] = A[i0:i1, :] @ B

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i0, i1) for (i0, i1) in chunks(n, rows_per_chunk)]
        for f in futs:
            f.result()

    return C


# ------------------------------
# Общие штуки для multiprocessing + memmap
# ------------------------------
_A_path = None
_B_path = None
_C_path = None
_N = None
_DTYPE = None
_BS = None

def _init_worker_memmap(a_path, b_path, c_path, n, dtype_str, bs):
    global _A_path, _B_path, _C_path, _N, _DTYPE, _BS
    _A_path = a_path
    _B_path = b_path
    _C_path = c_path
    _N = n
    _DTYPE = np.dtype(dtype_str)
    _BS = bs

def _open_A():
    return np.memmap(_A_path, mode="r", dtype=_DTYPE, shape=(_N, _N))

def _open_B():
    return np.memmap(_B_path, mode="r", dtype=_DTYPE, shape=(_N, _N))

def _open_C_write():
    return np.memmap(_C_path, mode="r+", dtype=_DTYPE, shape=(_N, _N))


def _worker_numpy_dot(task):
    i0, i1 = task
    A = _open_A()
    B = _open_B()
    C = _open_C_write()
    C[i0:i1, :] = A[i0:i1, :] @ B
    C.flush()
    return 1


def _worker_numba_blocked(task):
    i0, i1 = task
    A = _open_A()
    B = _open_B()
    C = _open_C_write()
    # важно: перед запуском C должен быть нулевой
    matmul_blocked_rows(A, B, C, i0, i1, _BS)
    C.flush()
    return 1


def _run_pool_memmap(A, B, worker_func, rows_per_chunk=200, processes=4, bs=64):
    """
    Запускает multiprocessing.Pool(spawn) и считает C через memmap.
    Это самый стабильный способ на macOS.
    """
    n = A.shape[0]
    dtype = A.dtype

    with tempfile.TemporaryDirectory() as tmp:
        a_path = os.path.join(tmp, "A.dat")
        b_path = os.path.join(tmp, "B.dat")
        c_path = os.path.join(tmp, "C.dat")

        A_mm = np.memmap(a_path, mode="w+", dtype=dtype, shape=(n, n))
        B_mm = np.memmap(b_path, mode="w+", dtype=dtype, shape=(n, n))
        C_mm = np.memmap(c_path, mode="w+", dtype=dtype, shape=(n, n))

        A_mm[:] = A
        B_mm[:] = B
        C_mm[:] = 0
        A_mm.flush(); B_mm.flush(); C_mm.flush()

        tasks = chunks(n, rows_per_chunk)

        ctx = mp.get_context("spawn")  # важно на macOS
        with ctx.Pool(
            processes=processes,
            initializer=_init_worker_memmap,
            initargs=(a_path, b_path, c_path, n, dtype.str, bs)
        ) as pool:
            pool.map(worker_func, tasks)

        # забираем результат в обычный numpy массив
        C_out = np.array(np.memmap(c_path, mode="r", dtype=dtype, shape=(n, n)))
        return C_out


def matmul_processes_numpy(A, B, rows_per_chunk=200, processes=4):
    return _run_pool_memmap(A, B, _worker_numpy_dot, rows_per_chunk=rows_per_chunk, processes=processes, bs=64)


def matmul_hybrid_mp_numba(A, B, rows_per_chunk=200, processes=4, bs=64):
    return _run_pool_memmap(A, B, _worker_numba_blocked, rows_per_chunk=rows_per_chunk, processes=processes, bs=bs)


# ------------------------------
# main benchmark
# ------------------------------
def main():
    n = 2000
    dtype = np.float32

    processes = 4
    threads = 8
    rows_per_chunk = 200
    bs = 64

    A, B = make_matrices(n=n, seed=1, dtype=dtype)

    # прогрев numba (чтобы компиляция не попала в замеры)
    a64 = np.random.rand(64, 64).astype(dtype)
    b64 = np.random.rand(64, 64).astype(dtype)
    c64 = np.zeros((64, 64), dtype=dtype)
    matmul_blocked_rows(a64, b64, c64, 0, 64, 16)

    t_threads = time_best(matmul_threads_numpy, A, B, rows_per_chunk=rows_per_chunk, workers=threads, repeats=2)
    t_proc = time_best(matmul_processes_numpy, A, B, rows_per_chunk=rows_per_chunk, processes=processes, repeats=2)
    t_hybrid = time_best(matmul_hybrid_mp_numba, A, B, rows_per_chunk=rows_per_chunk, processes=processes, bs=bs, repeats=2)

    print(f"n={n}, dtype={dtype}, rows_per_chunk={rows_per_chunk}, processes={processes}, threads={threads}, bs={bs}")
    print(f"pure threads (numpy @):        {t_threads:.3f} s")
    print(f"pure processes (numpy @):      {t_proc:.3f} s")
    print(f"hybrid (Pool + Numba blocked): {t_hybrid:.3f} s")


if __name__ == "__main__":
    # чтобы PyCharm на macOS не пытался форкать странно
    mp.set_start_method("spawn", force=True)
    main()
