import time
import math
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if radius is None:
        radius = int(math.ceil(3 * sigma))

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def convolve1d_edge(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    r = len(k) // 2

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode="edge")

    out = np.empty_like(arr, dtype=np.float64)

    if axis == 0:
        # по строкам (вертикально)
        for i in range(arr.shape[0]):
            window = padded[i : i + 2 * r + 1, :]
            # (2r+1, W) * (2r+1,) -> (W,)
            out[i, :] = (window * k[:, None]).sum(axis=0)
    elif axis == 1:
        # по столбцам (горизонтально)
        for j in range(arr.shape[1]):
            window = padded[:, j : j + 2 * r + 1]
            out[:, j] = (window * k[None, :]).sum(axis=1)
    else:
        raise ValueError("Only 2D arrays supported in this benchmark (axis 0 or 1).")

    return out


def gaussian_filter_2d_numpy(image: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel_1d(sigma)
    tmp = convolve1d_edge(image, k, axis=1)  # по X
    out = convolve1d_edge(tmp, k, axis=0)    # по Y
    return out


def _worker_gauss(args):
    img, sigma = args
    return gaussian_filter_2d_numpy(img, sigma)


def gaussian_filter_batch_parallel(images: list[np.ndarray], sigma: float, workers: int) -> list[np.ndarray]:
    # Важно: на macOS нужен __main__ guard, он ниже
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker_gauss, [(img, sigma) for img in images]))
    return results


def try_scipy_gaussian(image: np.ndarray, sigma: float):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=sigma, mode="nearest")
    except Exception as e:
        return None

def measure_time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def run_benchmark(
    img_sizes=(64, 128, 256),
    n_images=12,
    sigma=1.2,
    workers_list=(1, 2, 4, 8),
    repeats=3,
    seed=42,
):
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print("BENCHMARK: Gaussian filter (NumPy-separable) vs Parallel vs (optional) SciPy")
    print(f"sigma={sigma}, n_images={n_images}, repeats={repeats}")
    print("=" * 70)

    all_results = {}

    for size in img_sizes:
        H = W = size
        images = [rng.random((H, W), dtype=np.float64) for _ in range(n_images)]

        print(f"\n=== Image size: {H}x{W} ===")

        # Sequential batch
        def seq_run():
            for img in images:
                gaussian_filter_2d_numpy(img, sigma)

        t_seq = measure_time(seq_run, repeats=repeats)
        print(f"Sequential (our numpy): {t_seq:.4f} sec")

        # Parallel batch
        par_times = {}
        for w in workers_list:
            def par_run():
                gaussian_filter_batch_parallel(images, sigma, workers=w)

            t_par = measure_time(par_run, repeats=repeats)
            par_times[w] = t_par
            speedup = t_seq / t_par if t_par > 0 else float("nan")
            print(f"Parallel workers={w}: {t_par:.4f} sec | speedup: {speedup:.2f}x")

        # SciPy single-image (optional)
        scipy_out = try_scipy_gaussian(images[0], sigma)
        if scipy_out is None:
            print("SciPy gaussian_filter: SKIPPED (not available / error)")
            t_scipy = None
        else:
            def scipy_run():
                from scipy.ndimage import gaussian_filter
                for img in images:
                    gaussian_filter(img, sigma=sigma, mode="nearest")

            t_scipy = measure_time(scipy_run, repeats=repeats)
            print(f"SciPy gaussian_filter:     {t_scipy:.4f} sec (batch)")

        all_results[size] = {
            "t_seq": t_seq,
            "t_par": par_times,
            "t_scipy": t_scipy,
        }

    return all_results

def plot_speedup(results: dict, workers_list=(1, 2, 4, 8)):
    import matplotlib.pyplot as plt

    for size, data in results.items():
        t_seq = data["t_seq"]
        xs = []
        ys = []
        for w in workers_list:
            if w in data["t_par"]:
                xs.append(w)
                ys.append(t_seq / data["t_par"][w])

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title(f"Speedup vs workers (size {size}x{size})")
        plt.xlabel("Workers")
        plt.ylabel("Speedup (t_seq / t_parallel)")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    mp.freeze_support()

    # Подстрой под себя:
    results = run_benchmark(
        img_sizes=(64, 128, 256),   # можно поставить (32, 64) чтобы было быстрее
        n_images=8,                 # увеличь до 30-100 для более «тяжёлых» замеров
        sigma=1.2,
        workers_list=(1, 2, 4, 8),
        repeats=2,
        seed=1,
    )

    plot_speedup(results, workers_list=(1, 2, 4, 8))
