import numpy as np
import matplotlib.pyplot as plt


# ---------- генератор случайной ортонормальной матрицы ----------
def random_orthogonal(n: int, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    return Q


# ---------- симметричная SPD матрица с заданной обусловленностью ----------
def generate_spd_with_condition(n: int, kappa: float, seed=None) -> np.ndarray:
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")

    Q = random_orthogonal(n, seed=seed)

    # делаем спектр: от 1 до kappa
    # можно линейно или логарифмически: лог лучше, чтобы равномерно по порядкам
    eigvals = np.geomspace(1.0, kappa, num=n).astype(np.float64)

    A = Q @ np.diag(eigvals) @ Q.T
    # убираем численный мусор, чтобы было строго симметрично
    A = 0.5 * (A + A.T)
    return A


# ---------- Якоби для собственных значений ----------
def jacobi_eigvals_iterations(A0: np.ndarray, tol: float = 1e-8, max_iter: int = 20000):
    """
    Возвращает (eigvals, iters_used).
    Критерий остановки: max |a_ij| для i<j меньше tol.
    """
    A = A0.copy().astype(np.float64)
    n = A.shape[0]

    def max_offdiag(A):
        mx = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                v = abs(A[i, j])
                if v > mx:
                    mx = v
                    p, q = i, j
        return mx, p, q

    for it in range(max_iter):
        mx, p, q = max_offdiag(A)
        if mx < tol:
            return np.diag(A).copy(), it

        app, aqq, apq = A[p, p], A[q, q], A[p, q]

        if abs(apq) < 1e-30:
            continue

        theta = 0.5 * np.arctan2(2.0 * apq, (aqq - app))
        c = np.cos(theta)
        s = np.sin(theta)

        # обновляем строки/столбцы p,q
        for i in range(n):
            if i == p or i == q:
                continue
            aip, aiq = A[i, p], A[i, q]
            A[i, p] = c * aip - s * aiq
            A[p, i] = A[i, p]
            A[i, q] = s * aip + c * aiq
            A[q, i] = A[i, q]

        A[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0

    return np.diag(A).copy(), max_iter


# ---------- эксперимент: kappa vs итерации ----------
def run_experiment(n=30, kappas=None, trials=3, tol=1e-8, max_iter=20000, seed=1):
    if kappas is None:
        kappas = [1, 1e1, 1e2, 1e3, 1e4, 1e5]

    rng = np.random.default_rng(seed)
    results = []

    for k in kappas:
        its = []
        for _ in range(trials):
            s = int(rng.integers(1_000_000))
            A = generate_spd_with_condition(n, float(k), seed=s)

            # можно проверить реальную обусловленность (не обязательно)
            # cond_real = np.linalg.cond(A)

            _, it_used = jacobi_eigvals_iterations(A, tol=tol, max_iter=max_iter)
            its.append(it_used)

        results.append((float(k), float(np.mean(its)), float(np.std(its))))
    return results


def main():
    n = 30
    tol = 1e-8
    trials = 3
    kappas = [1, 10, 1e2, 1e3, 1e4, 1e5]

    results = run_experiment(n=n, kappas=kappas, trials=trials, tol=tol, seed=7)

    print("kappa -> iterations (mean ± std)")
    for k, m, s in results:
        print(f"{k:>8.0f} -> {m:8.1f} ± {s:.1f}")

    # график
    ks = [r[0] for r in results]
    it_mean = [r[1] for r in results]
    it_std = [r[2] for r in results]

    plt.figure()
    plt.xscale("log")
    plt.errorbar(ks, it_mean, yerr=it_std, fmt="o-", capsize=4)
    plt.xlabel("Condition number (kappa)")
    plt.ylabel("Jacobi iterations to converge")
    plt.title("Condition number vs iterations (Jacobi eigenvalues)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
