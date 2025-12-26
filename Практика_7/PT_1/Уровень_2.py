import ast
from collections import defaultdict


class LoopInfo:
    def __init__(self, lineno_start, lineno_end):
        self.start = lineno_start
        self.end = lineno_end
        self.reads = defaultdict(list)
        self.writes = defaultdict(list)
        self.has_dependency = False


class LoopDoctor(ast.NodeVisitor):
    def __init__(self):
        self.loops = []

    def visit_For(self, node):
        loop = LoopInfo(node.lineno, node.end_lineno)

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                self._handle_assign(child, loop)

        self._analyze_dependencies(loop)
        self.loops.append(loop)
        self.generic_visit(node)

    def _handle_assign(self, node, loop):
        # Левая часть — запись
        if isinstance(node.targets[0], ast.Subscript):
            name, index = self._parse_subscript(node.targets[0])
            loop.writes[name].append(index)

        # Правая часть — чтение
        for child in ast.walk(node.value):
            if isinstance(child, ast.Subscript):
                name, index = self._parse_subscript(child)
                loop.reads[name].append(index)

    def _parse_subscript(self, node):
        name = node.value.id
        if isinstance(node.slice, ast.BinOp):
            return name, ast.unparse(node.slice)
        return name, ast.unparse(node.slice)

    def _analyze_dependencies(self, loop):
        for var in loop.writes:
            for w in loop.writes[var]:
                for r in loop.reads.get(var, []):
                    if r != w:
                        loop.has_dependency = True

    def report(self):
        print("Анализ завершён!")
        print(f"Найдено {len(self.loops)} цикла:\n")

        for i, loop in enumerate(self.loops, 1):
            print(f"ЦИКЛ {i} (строки {loop.start}-{loop.end}):")

            if loop.has_dependency:
                print("Проблема: Обнаружена зависимость данных.")
                print("Советы:")
                print(" [✓] Выполни зависимую операцию в отдельном цикле.")
                print(" [ ] Не пытайся сливать этот цикл с другими.")
            else:
                print("Проблем: зависимостей нет, цикл чистый.")
                print("Советы:")
                print(" [✓] Отличный кандидат для объединения с соседним циклом.")
                print(" [✓] Можно применить @jit(nopython=True).")

            print()


def analyze_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    analyzer = LoopDoctor()
    analyzer.visit(tree)
    analyzer.report()


if __name__ == "__main__":
    analyze_file("my_code.py")
