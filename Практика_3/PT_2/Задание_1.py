import threading

def compute_chunk(old, new, start, end):
    for i in range(max(2, start), end):
        new[i] = old[i-2] * 2 + old[i-1] * 0.5

n = 1000
a = [i for i in range(n)]

old = a.copy()
new = a.copy()

num_threads = 4
threads = []

chunk = n // num_threads

for t in range(num_threads):
    start = t * chunk
    end = (t+1) * chunk if t < num_threads - 1 else n
    thread = threading.Thread(target=compute_chunk, args=(old, new, start, end))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

a = new
print("Пример результата:", a[:10])
