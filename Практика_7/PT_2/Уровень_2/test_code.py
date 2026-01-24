def process_data(a):
    s = 0
    for i in range(len(a)):
        for j in range(len(a)):
            s += a[i] * a[j]
    return s
