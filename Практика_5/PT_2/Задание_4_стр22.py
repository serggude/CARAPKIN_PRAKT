import numpy as np
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def convolution_original(image, kernel):
    height, width = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output

def convolution_blocked(image, kernel, block_size=32):
    height, width = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image)

    for ii in range(0, height, block_size):
        for jj in range(0, width, block_size):
            for i in range(ii, min(ii + block_size, height)):
                for j in range(jj, min(jj + block_size, width)):
                    region = padded[i:i + k_h, j:j + k_w]
                    output[i, j] = np.sum(region * kernel)

    return output

def _convolution_worker(args):
    padded, kernel, row_range, width, k_h, k_w = args
    out = np.zeros((len(row_range), width))

    for idx, i in enumerate(row_range):
        for j in range(width):
            region = padded[i:i + k_h, j:j + k_w]
            out[idx, j] = np.sum(region * kernel)

    return row_range.start, out

def convolution_parallel(image, kernel, num_workers=None):
    height, width = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image)

    workers = num_workers or multiprocessing.cpu_count()
    chunk = height // workers

    tasks = []
    for w in range(workers):
        start = w * chunk
        end = height if w == workers - 1 else (w + 1) * chunk
        tasks.append((padded, kernel, range(start, end), width, k_h, k_w))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start_row, block in executor.map(_convolution_worker, tasks):
            output[start_row:start_row + block.shape[0], :] = block

    return output

def measure(func, *args):
    t0 = time.perf_counter()
    res = func(*args)
    return res, time.perf_counter() - t0


def run_experiment():
    sizes = [(256, 256), (512, 512)]
    kernels = [3, 5, 7]

    for h, w in sizes:
        print(f"\n=== Image size: {h}x{w} ===")
        image = np.random.rand(h, w)

        for k in kernels:
            kernel = np.random.rand(k, k)
            print(f"\nKernel size: {k}x{k}")

            _, t_orig = measure(convolution_original, image, kernel)
            _, t_block = measure(convolution_blocked, image, kernel)
            _, t_par = measure(convolution_parallel, image, kernel)

            print(f"Original:   {t_orig:.4f} сек")
            print(f"Blocked:    {t_block:.4f} сек")
            print(f"Parallel:   {t_par:.4f} сек")

            if SCIPY_AVAILABLE:
                _, t_scipy = measure(convolve2d, image, kernel, "same")
                print(f"SciPy:      {t_scipy:.4f} сек")


if __name__ == "__main__":
    run_experiment()
