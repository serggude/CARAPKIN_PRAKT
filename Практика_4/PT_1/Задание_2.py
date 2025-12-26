def dependency_analyzer(python_code):
    dependencies = []

    last_write = {}  # переменная -> номер строки последней записи
    last_read = {}   # переменная -> номер строки последнего чтения

    lines = python_code.strip().split("\n")

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        if not line or "=" not in line:
            continue

        left, right = line.split("=", 1)
        left = left.strip()

        right_vars = [
            token.strip()
            for token in right.replace("+", " ").replace("-", " ")
                             .replace("*", " ").replace("/", " ").split()
            if token.isidentifier()
        ]

        # FLOW
        for var in right_vars:
            if var in last_write:
                dependencies.append({
                    "type": "FLOW",
                    "from": last_write[var],
                    "to": line_num,
                    "variable": var
                })
            last_read[var] = line_num

        # OUTPUT
        if left in last_write:
            dependencies.append({
                "type": "OUTPUT",
                "from": last_write[left],
                "to": line_num,
                "variable": left
            })

        # ANTI
        if left in last_read:
            dependencies.append({
                "type": "ANTI",
                "from": last_read[left],
                "to": line_num,
                "variable": left
            })

        last_write[left] = line_num

    return dependencies

test_code = """
a = x * 2
b = y + 10
c = a + b
a = c * 3
"""

deps = dependency_analyzer(test_code)

for d in deps:
    print(d)
