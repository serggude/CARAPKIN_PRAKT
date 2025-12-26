import ast
import concurrent.futures
import multiprocessing
from typing import Dict, List


class LoopAnalyzer:
    # Анализ зависимостей (упрощённый AST-анализ)
    def analyze_dependencies(self, code: str) -> Dict:
        tree = ast.parse(code)

        reads = []
        writes = []

        class Visitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                # левая часть — запись
                for t in node.targets:
                    if isinstance(t, ast.Subscript):
                        writes.append(ast.unparse(t))
                # правая часть — чтение
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Subscript):
                        reads.append(ast.unparse(child))
                self.generic_visit(node)

        Visitor().visit(tree)

        dep_type = "D=0"

        for r in reads:
            if any(r.split("[")[0] in w for w in writes):
                dep_type = "D=1"

        arrays = list(set([x.split("[")[0] for x in reads + writes]))

        return {
            "type": dep_type,
            "reads": reads,
            "writes": writes,
            "arrays": arrays
        }
    def recommend_transformations(self, dependencies: Dict) -> List[str]:
        recommendations = []

        if dependencies["type"] == "D=1":
            recommendations.append("Красно-черное упорядочение")
            recommendations.append("Сдваивание буферов")

        elif len(dependencies["arrays"]) > 1:
            recommendations.append("Разделение циклов")
            recommendations.append("Комбинированное преобразование")

        else:
            recommendations.append("Параллельный for")

        return recommendations

    def apply_double_buffering(self, func, n, arr):
        old = arr.copy()
        new = arr.copy()

        for i in range(1, n):
            new[i] = old[i - 1] * 2 + i * 3

        return new

    def parallel_execute(self, func, n, arr):
        old = arr.copy()
        new = arr.copy()

        def worker(i):
            return i, old[i - 1] * 2 + i * 3

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count()
        ) as executor:
            for i, val in executor.map(worker, range(1, n)):
                new[i] = val

        return new

if __name__ == "__main__":
    analyzer = LoopAnalyzer()

    code1 = """
for i in range(1, n):
    a[i] = a[i-1] * 2 + i
"""

    code2 = """
for i in range(n):
    a[i] = b[i] + c[i]
"""

    code3 = """
for i in range(1, n):
    x[i] = y[i-1] + z[i-1]
    y[i] = x[i-1] * 2
"""

    for idx, code in enumerate([code1, code2, code3], start=1):
        print("=" * 60)
        print(f"ПРИМЕР {idx}")
        deps = analyzer.analyze_dependencies(code)
        print("Зависимости:", deps)
        print("Рекомендации:", analyzer.recommend_transformations(deps))

    n = 10_000
    a = [1] * n

    print("\n=== Автоматическое преобразование + распараллеливание ===")
    res = analyzer.parallel_execute(None, n, a)
    print("Готово, результат вычислен")
