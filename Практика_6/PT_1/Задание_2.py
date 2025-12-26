import re


class NestedLoopAnalyzer:
    def __init__(self):
        self.loop_vars = []          # ['i', 'j']
        self.array_accesses = []     # все чтения/записи
        self.dependencies = []       # найденные зависимости

    # -------------------------------
    # 1. Парсинг структуры циклов
    # -------------------------------
    def parse_loop_structure(self, code_string):
        self.loop_vars = re.findall(r'for\s+(\w+)\s+in\s+range', code_string)

    # -------------------------------
    # 2. Анализ обращений к массивам
    # -------------------------------
    def analyze_array_accesses(self, code_string):
        self.array_accesses = []

        lines = code_string.splitlines()

        for line in lines:
            line = line.strip()
            if not line or '=' not in line:
                continue

            lhs, rhs = line.split('=', 1)

            # запись
            for m in re.finditer(r'(\w+)\[([^\]]+)\]\[([^\]]+)\]', lhs):
                self.array_accesses.append({
                    'array': m.group(1),
                    'operation': 'write',
                    'indices': (m.group(2), m.group(3))
                })

            # чтение
            for m in re.finditer(r'(\w+)\[([^\]]+)\]\[([^\]]+)\]', rhs):
                self.array_accesses.append({
                    'array': m.group(1),
                    'operation': 'read',
                    'indices': (m.group(2), m.group(3))
                })

    # -------------------------------
    # 3. Поиск зависимостей
    # -------------------------------
    def find_dependencies(self):
        self.dependencies = []

        for w in self.array_accesses:
            if w['operation'] != 'write':
                continue

            for r in self.array_accesses:
                if r['operation'] != 'read':
                    continue

                if w['array'] != r['array']:
                    continue

                if w['indices'] == r['indices']:
                    dep_type = 'Flow'
                else:
                    dep_type = 'Potential'

                self.dependencies.append({
                    'array': w['array'],
                    'source': w['indices'],
                    'sink': r['indices'],
                    'type': dep_type
                })

    # -------------------------------
    # 4. Векторы направлений
    # -------------------------------
    def compute_direction_vectors(self):
        for dep in self.dependencies:
            src_i, src_j = dep['source']
            dst_i, dst_j = dep['sink']

            def direction(a, b):
                if a == b:
                    return '='
                if '-' in a:
                    return '<'
                if '+' in a:
                    return '>'
                return '?'

            dep['direction_vector'] = (
                direction(src_i, dst_i),
                direction(src_j, dst_j)
            )

            dep['recommendation'] = (
                "Сдваивание буферов" if dep['type'] == 'Flow'
                else "Переупорядочивание итераций"
            )

    # -------------------------------
    # 5. Генерация отчёта
    # -------------------------------
    def generate_report(self):
        report = "📊 ОТЧЕТ АНАЛИЗАТОРА ЦИКЛОВ\n"
        report += "=" * 50 + "\n\n"

        if not self.dependencies:
            return report + "✅ Зависимости не обнаружены\n"

        for dep in self.dependencies:
            report += f"🔗 Зависимость массива `{dep['array']}`\n"
            report += f"   Источник: {dep['source']}\n"
            report += f"   Сток:     {dep['sink']}\n"
            report += f"   Вектор направлений: {dep['direction_vector']}\n"
            report += f"   Тип: {dep['type']}\n"
            report += f"   💡 Рекомендация: {dep['recommendation']}\n\n"

        return report

    # -------------------------------
    # Главный метод
    # -------------------------------
    def analyze_loop(self, code_string):
        self.parse_loop_structure(code_string)
        self.analyze_array_accesses(code_string)
        self.find_dependencies()
        self.compute_direction_vectors()
        return self.generate_report()

code = """
for i in range(1, n):
    for j in range(1, m):
        a[i][j] = a[i-1][j] + b[i][j-1]
        b[i][j] = a[i][j] * 2
"""

analyzer = NestedLoopAnalyzer()
report = analyzer.analyze_loop(code)
print(report)
