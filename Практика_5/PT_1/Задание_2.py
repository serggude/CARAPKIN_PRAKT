import re

def dependency_analyzer(loop_code: str, unroll_iters: int = 4):
    # --- парсинг заголовка цикла ---
    header = re.search(
        r"for\s+(?P<i>\w+)\s+in\s+range\(\s*(?P<start>\d+)\s*,\s*(?P<end>\w+)\s*\)\s*:",
        loop_code
    )
    if not header:
        raise ValueError("Ожидается цикл вида: for i in range(k, n):")

    it = header.group("i")
    start = int(header.group("start"))

    lines = [l.strip() for l in loop_code.splitlines() if l.strip()]
    body = lines[1:]

    write_re = re.compile(rf"(?P<a>\w+)\[{it}\]\s*=\s*(?P<rhs>.+)")
    read_re = re.compile(rf"(?P<a>\w+)\[{it}(?P<op>[+-])?(?P<k>\d+)?\]")

    writes = []
    for line in body:
        m = write_re.match(line)
        if m:
            writes.append((m.group("a"), m.group("rhs")))

    iters = list(range(start, start + unroll_iters))

    # --- развёртка ---
    unrolled = []
    for i in iters:
        for arr, rhs in writes:
            def repl(m):
                k = int(m.group("k")) if m.group("k") else 0
                if not m.group("op"):
                    idx = i
                elif m.group("op") == "+":
                    idx = i + k
                else:
                    idx = i - k
                return f"{m.group('a')}[{idx}]"

            rhs_i = read_re.sub(repl, rhs)
            unrolled.append(f"i={i}: {arr}[{i}] = {rhs_i}")

    # --- карта записей ---
    write_map = {}
    for i in iters:
        for arr, _ in writes:
            write_map[(arr, i)] = i

    # --- зависимости ---
    deps = []
    max_dist = 0

    for i in iters:
        for arr, rhs in writes:
            def repl2(m):
                k = int(m.group("k")) if m.group("k") else 0
                if not m.group("op"):
                    idx = i
                elif m.group("op") == "+":
                    idx = i + k
                else:
                    idx = i - k
                return f"{m.group('a')}[{idx}]"

            rhs_i = read_re.sub(repl2, rhs)

            for rm in re.finditer(r"(\w+)\[(\-?\d+)\]", rhs_i):
                a, idx = rm.group(1), int(rm.group(2))
                if (a, idx) in write_map and i > idx:
                    d = i - idx
                    max_dist = max(max_dist, d)
                    deps.append({
                        "type": "FLOW",
                        "from": idx,
                        "to": i,
                        "variable": a,
                        "distance": d
                    })

    deps = { (d["from"], d["to"], d["variable"]): d for d in deps }.values()

    parallelizable = True
    if deps:
        min_d = min(d["distance"] for d in deps)
        if min_d == 1:
            parallelizable = False

    return {
        "dependencies": list(deps),
        "max_distance": max_dist,
        "parallelizable": parallelizable,
        "unrolled_code": "\n".join(unrolled)
    }

def run_test(name, loop_code):
    print("=" * 80)
    print(f"ТЕСТ: {name}")
    print("-" * 80)
    print(loop_code.strip())
    print("-" * 80)

    result = dependency_analyzer(loop_code)

    print("UNROLLED CODE:")
    print(result["unrolled_code"])
    print()

    print("DEPENDENCIES:")
    if not result["dependencies"]:
        print("Нет зависимостей")
    else:
        for d in result["dependencies"]:
            print(d)

    print()
    print("MAX DISTANCE:", result["max_distance"])
    print("PARALLELIZABLE:", result["parallelizable"])
    print()


# ===== ТЕСТ 1: D = 1 (критический) =====
run_test(
    "Critical case (D=1)",
    """
for i in range(1, n):
    a[i] = a[i-1] + 1
"""
)


# ===== ТЕСТ 2: D = 2 (частично параллелизуемый) =====
run_test(
    "Distance 2 case (D=2)",
    """
for i in range(2, n):
    a[i] = a[i-2] * 2
"""
)


# ===== ТЕСТ 3: Несколько чтений =====
run_test(
    "Multiple reads",
    """
for i in range(2, n):
    a[i] = a[i-1] + a[i-2]
"""
)


# ===== ТЕСТ 4: Без зависимостей =====
run_test(
    "No dependencies",
    """
for i in range(0, n):
    a[i] = b[i] + 1
"""
)


# ===== ТЕСТ 5: Смешанные массивы =====
run_test(
    "Mixed arrays",
    """
for i in range(1, n):
    a[i] = b[i-1] + c[i]
"""
)
