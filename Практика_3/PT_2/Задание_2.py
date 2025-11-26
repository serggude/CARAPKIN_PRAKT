import threading
import numpy as np

height = 100
width = 100
image = np.random.randint(0, 255, (6, 6)).astype(float)

print("Исходное изображение:")
print(image)

old = image.copy()

def blur_pixel(src, i, j):
    return (
        src[i-1, j] + src[i+1, j] +
        src[i, j-1] + src[i, j+1] +
        src[i, j]
    ) / 5


def process_chunk(src, dst, start_i, end_i, color):
    for i in range(start_i, end_i):
        for j in range(1, width - 1):
            if (i + j) % 2 == color:   
                dst[i, j] = blur_pixel(src, i, j)


def blur_pixel(src, i, j):
    return (
        src[i-1, j] + src[i+1, j] +
        src[i, j-1] + src[i, j+1] +
        src[i, j]
    ) / 5


def red_black_blur(image, num_threads=4):
    height, width = image.shape

    src = image.copy()
    dst = image.copy()

    def process_chunk(start_i, end_i, color):
        for i in range(start_i, end_i):
            for j in range(1, width - 1):
                if (i + j) % 2 == color:
                    dst[i, j] = blur_pixel(src, i, j)

    # красные клетки
    threads = []
    inner_rows = max(0, height - 2)
    chunk = max(1, inner_rows // num_threads) if inner_rows > 0 else 1

    for t in range(num_threads):
        start = 1 + t * chunk
        end = 1 + (t + 1) * chunk
        if t == num_threads - 1:
            end = height - 1
        if start >= height - 1:
            break
        thread = threading.Thread(target=process_chunk, args=(start, end, 0))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    src, dst = dst, src

    # черные клетки
    threads = []

    def process_chunk_black(start_i, end_i):
        for i in range(start_i, end_i):
            for j in range(1, width - 1):
                if (i + j) % 2 == 1:  # черные
                    dst[i, j] = blur_pixel(src, i, j)

    for t in range(num_threads):
        start = 1 + t * chunk
        end = 1 + (t + 1) * chunk
        if t == num_threads - 1:
            end = height - 1
        if start >= height - 1:
            break
        thread = threading.Thread(target=process_chunk_black, args=(start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return dst



result = red_black_blur(image.copy(), num_threads=4)

print("\nРазмытие:")
print(result)
