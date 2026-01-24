import numpy as np
from concurrent.futures import ThreadPoolExecutor


def make_matrix(n=10, seed=1):
    rng = np.random.default_rng(seed)
    D = rng.uniform(1, 9, size=(n, n)).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return D


def fw_usual(D0):
    """Обычный Флойд–Уоршелл (для проверки)."""
    D = D0.copy()
    n = D.shape[0]
    for k in range(n):
        row_k = D[k, :].copy()
        for i in range(n):
            cand = D[i, k] + row_k
            D[i, :] = np.minimum(D[i, :], cand)
    return D


def update_block_parallel(block, left_vec, top_vec, workers=4):
    """
    Обновляет один блок по формуле:
        block[r, c] = min(block[r, c], left_vec[r] + top_vec[c])

    Здесь параллелим по строкам блока (r).
    """
    h = block.shape[0]

    def work(r0, r1):
        for r in range(r0, r1):
            cand = left_vec[r] + top_vec
            block[r, :] = np.minimum(block[r, :], cand)

    step = max(1, (h + workers - 1) // workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for r0 in range(0, h, step):
            r1 = min(h, r0 + step)
            futures.append(ex.submit(work, r0, r1))
        for f in futures:
            f.result()


def blocked_fw(D0, bs=3, workers=4):
    """
    Блочный Флойд–Уоршелл:
    - делим матрицу на блоки bs x bs
    - обрабатываем блоки по фазам (как в blocked FW)
    - параллелизм делаем внутри обработки одного блока (по строкам блока)

    На маленькой 10x10 матрице это просто демонстрация идеи.
    В реальном "экономном по памяти" варианте блоки можно подгружать/выгружать.
    """
    D = D0.copy()
    n = D.shape[0]
    nb = (n + bs - 1) // bs

    def sl(b):
        a = b * bs
        b2 = min(n, (b + 1) * bs)
        return slice(a, b2)

    for bk in range(nb):
        sk = sl(bk)

        # ---- Фаза 1: диагональный блок (bk, bk) обычным FW внутри блока
        Dkk = D[sk, sk]
        m = Dkk.shape[0]
        for k in range(m):
            row_k = Dkk[k, :].copy()
            for i in range(m):
                cand = Dkk[i, k] + row_k
                Dkk[i, :] = np.minimum(Dkk[i, :], cand)

        # ---- Фаза 2: обновляем блоки в строке (bk, bj), bj != bk
        for bj in range(nb):
            if bj == bk:
                continue
            sj = sl(bj)
            Dkj = D[sk, sj]

            # Dkj = min(Dkj, Dkk[:,t] + Dkj[t,:]) по t внутри диагонального блока
            for t in range(m):
                left = Dkk[:, t]
                top = Dkj[t, :].copy()
                update_block_parallel(Dkj, left, top, workers=workers)

        # ---- Фаза 2: обновляем блоки в столбце (bi, bk), bi != bk
        for bi in range(nb):
            if bi == bk:
                continue
            si = sl(bi)
            Dik = D[si, sk]
            hi = Dik.shape[0]

            # Dik = min(Dik, Dik[:,t] + Dkk[t,:]) по t
            for t in range(m):
                left = Dik[:, t]          # длина hi
                top = Dkk[t, :].copy()    # длина m
                update_block_parallel(Dik, left, top, workers=workers)

        # ---- Фаза 3: все остальные блоки (bi, bj), bi != bk, bj != bk
        for bi in range(nb):
            if bi == bk:
                continue
            si = sl(bi)
            for bj in range(nb):
                if bj == bk:
                    continue
                sj = sl(bj)

                Dij = D[si, sj]
                Dik = D[si, sk]
                Dkj = D[sk, sj]

                # Dij = min(Dij, Dik[:,t] + Dkj[t,:]) по t
                for t in range(m):
                    left = Dik[:, t]          # длина hi
                    top = Dkj[t, :].copy()    # длина wj
                    update_block_parallel(Dij, left, top, workers=workers)

    return D


def main():
    D0 = make_matrix(n=10, seed=1)

    D1 = fw_usual(D0)
    D2 = blocked_fw(D0, bs=3, workers=4)

    print("same result:", np.allclose(D1, D2, atol=1e-9, rtol=0.0))
    print("example:", D1[0, 7], D2[0, 7])


if __name__ == "__main__":
    main()
