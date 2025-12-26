def homework_expert():
    def analyze_and_suggest(code_string):
        suggestions = []

        last_write = {}
        last_read = {}

        lines = code_string.strip().split("\n")

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            if not line or "=" not in line:
                continue

            left, right = line.split("=", 1)
            left = left.strip()

            right_vars = [
                token.strip()
                for token in right.replace("+", " ")
                                  .replace("-", " ")
                                  .replace("*", " ")
                                  .replace("/", " ")
                                  .split()
                if token.isidentifier()
            ]

            # FLOW (RAW)
            for var in right_vars:
                if var in last_write:
                    suggestions.append({
                        "line": line_num,
                        "type": "FLOW",
                        "variable": var,
                        "problem": f"Чтение '{var}' после записи",
                        "solution": "Использовать double buffering или вынести запись в отдельный буфер",
                        "example_fix": f"# Читать '{var}' из старого буфера, писать в новый"
                    })
                last_read[var] = line_num

            # OUTPUT (WAW)
            if left in last_write:
                suggestions.append({
                    "line": line_num,
                    "type": "OUTPUT",
                    "variable": left,
                    "problem": f"Повторная запись в '{left}'",
                    "solution": "Переименовать переменные и разделить каналы данных",
                    "example_fix": f"{left}_new = ...  # вместо перезаписи '{left}'"
                })

            # ANTI (WAR)
            if left in last_read:
                suggestions.append({
                    "line": line_num,
                    "type": "ANTI",
                    "variable": left,
                    "problem": f"Запись '{left}' после чтения",
                    "solution": "Создать копию данных перед чтением",
                    "example_fix": f"{left}_snapshot = {left}  # копия для безопасного чтения"
                })

            last_write[left] = line_num

        return suggestions

    return analyze_and_suggest

analyzer = homework_expert()


def run_test(name, code):
    print("=" * 80)
    print(f"ТЕСТ: {name}")
    print("-" * 80)
    print(code.strip())
    print("-" * 80)

    results = analyzer(code)

    if not results:
        print("Зависимости не обнаружены")
    else:
        for r in results:
            print(r)
    print()


# ТЕСТ 1. Output зависимость (два стримера на одном канале)
run_test(
    "Output dependency (WAW)",
    """
result = x + y
result = process_data(z)
"""
)


# ТЕСТ 2. Anti зависимость (прочитал — потом перезаписал)
run_test(
    "Anti dependency (WAR)",
    """
user_name = user_data['name']
user_email = user_data['email']
user_data = new_settings
"""
)


# ТЕСТ 3. Flow + Output + Anti (пример из части 1)
run_test(
    "Flow + Output + Anti (комбинированный пример)",
    """
a = x * 2
b = y + 10
c = a + b
a = c * 3
"""
)


#  ТЕСТ 4. Наивный фильтр размытия (Flow зависимости в массиве)
run_test(
    "Flow dependency (blur filter, in-place)",
    """
pixels[i][j] = (
    pixels[i-1][j] +
    pixels[i+1][j] +
    pixels[i][j-1] +
    pixels[i][j+1] +
    pixels[i][j]
) // 5
"""
)


# ТЕСТ 5. Double buffering (зависимости устранены)
run_test(
    "Double buffering (зависимости устранены)",
    """
new_pixels[i][j] = (
    old_pixels[i-1][j] +
    old_pixels[i+1][j] +
    old_pixels[i][j-1] +
    old_pixels[i][j+1] +
    old_pixels[i][j]
) // 5
"""
)


#  ТЕСТ 6. Anti зависимость в обработке потоков
run_test(
    "Anti dependency (data stream processing)",
    """
stats = analyze_chunk(data_chunk)
data_chunk = transform(data_chunk)
"""
)


# ТЕСТ 7. Медианный фильтр (наивный, in-place)
run_test(
    "Median filter (in-place, Flow dependency)",
    """
neighborhood = []
neighborhood.append(image[i+1][j])
neighborhood.append(image[i][j])
image[i][j] = sorted(neighborhood)[0]
"""
)


# ТЕСТ 8. Медианный фильтр (double buffering)
run_test(
    "Median filter (double buffering, OK)",
    """
next_image[i][j] = median(current[i-1][j], current[i][j], current[i+1][j])
"""
)


# ТЕСТ 9. Красно-чёрное упорядочение (две фазы)
run_test(
    "Red-Black ordering (двухфазная схема)",
    """
if (i + j) % 2 == 0:
    new_grid[i][j] = grid[i-1][j] + grid[i+1][j]
else:
    new_grid[i][j] = new_grid[i-1][j] + new_grid[i+1][j]
"""
)



