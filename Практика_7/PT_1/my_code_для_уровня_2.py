for i in range(n):
    a[i] = a[i - 1] + 1

for i in range(n):
    b[i] = a[i] * 2

for i in range(n):
    c[i] = b[i] + 5
