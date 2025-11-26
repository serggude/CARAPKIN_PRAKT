import matplotlib.pyplot as plt
import numpy as np

def amdahl_speedup(P, N):
    return 1 / ((1 - P) + P / N)

N_values = np.arange(1, 17)

P_values = [0.5, 0.8, 0.95, 0.99]

colors = ["blue", "green", "red", "purple"]

plt.figure(figsize=(12, 6))

for P, color in zip(P_values, colors):
    S_values = [amdahl_speedup(P, N) for N in N_values]
    plt.plot(N_values, S_values, marker="o", color=color, label=f"P = {P}")

plt.title("Теоретическое ускорение по закону Амдала", fontsize=16)
plt.xlabel("Число потоков (N)", fontsize=12)
plt.ylabel("Ускорение S(N)", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(N_values)
plt.show()
