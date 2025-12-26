import numpy as np

def non_local_means_denoising(image, search_window=3, patch_size=3, h=10.0):
    height, width = image.shape
    pad_s = search_window // 2
    pad_p = patch_size // 2

    padded = np.pad(image, pad_s + pad_p, mode='reflect')
    output = np.zeros_like(image, dtype=float)

    for i in range(height):
        for j in range(width):
            i0 = i + pad_s + pad_p
            j0 = j + pad_s + pad_p

            ref_patch = padded[
                i0 - pad_p:i0 + pad_p + 1,
                j0 - pad_p:j0 + pad_p + 1
            ]

            w_sum = 0.0
            pixel_sum = 0.0

            for di in range(-pad_s, pad_s + 1):
                for dj in range(-pad_s, pad_s + 1):
                    ii = i0 + di
                    jj = j0 + dj

                    patch = padded[
                        ii - pad_p:ii + pad_p + 1,
                        jj - pad_p:jj + pad_p + 1
                    ]

                    dist2 = np.sum((ref_patch - patch) ** 2)
                    w = np.exp(-dist2 / (h * h))

                    w_sum += w
                    pixel_sum += w * padded[ii, jj]

            output[i, j] = pixel_sum / w_sum

    return output

from concurrent.futures import ProcessPoolExecutor
import os

def _nlm_worker(args):
    image, i, j, search_window, patch_size, h = args
    height, width = image.shape

    pad_s = search_window // 2
    pad_p = patch_size // 2
    padded = np.pad(image, pad_s + pad_p, mode='reflect')

    i0 = i + pad_s + pad_p
    j0 = j + pad_s + pad_p

    ref_patch = padded[
        i0 - pad_p:i0 + pad_p + 1,
        j0 - pad_p:j0 + pad_p + 1
    ]

    w_sum = 0.0
    pixel_sum = 0.0

    for di in range(-pad_s, pad_s + 1):
        for dj in range(-pad_s, pad_s + 1):
            ii = i0 + di
            jj = j0 + dj

            patch = padded[
                ii - pad_p:ii + pad_p + 1,
                jj - pad_p:jj + pad_p + 1
            ]

            dist2 = np.sum((ref_patch - patch) ** 2)
            w = np.exp(-dist2 / (h * h))

            w_sum += w
            pixel_sum += w * padded[ii, jj]

    return i, j, pixel_sum / w_sum


def non_local_means_parallel(image, search_window=3, patch_size=3, h=10.0, workers=4):
    height, width = image.shape
    output = np.zeros_like(image, dtype=float)

    tasks = [
        (image, i, j, search_window, patch_size, h)
        for i in range(height)
        for j in range(width)
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, j, value in executor.map(_nlm_worker, tasks):
            output[i, j] = value

    return output

import time

def benchmark_nlm():
    image = np.random.rand(64, 64)

    t0 = time.time()
    non_local_means_denoising(image)
    t_seq = time.time() - t0

    results = {"sequential": t_seq}

    for p in [1, 2, 4, 8]:
        t0 = time.time()
        non_local_means_parallel(image, workers=p)
        results[p] = time.time() - t0

    return results

import time
import numpy as np
import multiprocessing as mp

def run_tests():
    print("=" * 60)
    print("ТЕСТ NON-LOCAL MEANS DENOISING")
    print("=" * 60)

    image_sizes = [(32, 32), (64, 64)]
    workers_list = [1, 2, 4, 8]

    for shape in image_sizes:
        print(f"\n=== Размер изображения: {shape[0]}x{shape[1]} ===")

        image = np.random.rand(*shape)

        # Последовательная версия
        t0 = time.time()
        result_seq = non_local_means_denoising(
            image,
            search_window=3,
            patch_size=3,
            h=10.0
        )
        t_seq = time.time() - t0

        print(f"Последовательная версия: {t_seq:.4f} сек")

        # Параллельные версии
        for workers in workers_list:
            t0 = time.time()
            result_par = non_local_means_parallel(
                image,
                search_window=3,
                patch_size=3,
                h=10.0,
                workers=workers
            )
            t_par = time.time() - t0

            # Проверка корректности
            correct = np.allclose(result_seq, result_par, atol=1e-6)

            speedup = t_seq / t_par if t_par > 0 else 0.0

            print(
                f"{workers} процессов | "
                f"Время: {t_par:.4f} сек | "
                f"Ускорение: {speedup:.2f}x | "
                f"Корректность: {correct}"
            )


if __name__ == "__main__":
    mp.freeze_support()  # важно для PyCharm / macOS
    run_tests()
