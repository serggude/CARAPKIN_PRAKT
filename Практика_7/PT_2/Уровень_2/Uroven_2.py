import ast
import argparse


class LoopOptimizerLite:
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        self.loops = []

    # ---------- АНАЛИЗ ----------

    def analyze(self):
        self._find_loops(self.tree)
        return self._make_report()

    def _find_loops(self, node, depth=0):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.For):
                loop_info = {
                    "type": "for",
                    "line": child.lineno,
                    "depth": depth + 1,
                    "var": self._get_loop_var(child),
                }
                self.loops.append(loop_info)
                self._find_loops(child, depth + 1)

            elif isinstance(child, ast.While):
                loop_info = {
                    "type": "while",
                    "line": child.lineno,
                    "depth": depth + 1,
                    "var": None,
                }
                self.loops.append(loop_info)
                self._find_loops(child, depth + 1)

            else:
                self._find_loops(child, depth)

    def _get_loop_var(self, node):
        if isinstance(node.target, ast.Name):
            return node.target.id
        return "complex"

    # ---------- ОТЧЁТ ----------

    def _make_report(self):
        report = []
        report.append("Анализ циклов")
        report.append("=" * 40)

        if not self.loops:
            report.append("Циклы не найдены.")
            return "\n".join(report)

        max_depth = max(loop["depth"] for loop in self.loops)

        report.append(f"Найдено циклов: {len(self.loops)}")
        report.append(f"Максимальная вложенность: {max_depth}")
        report.append("")

        report.append("Найденные циклы:")
        for loop in self.loops:
            report.append(
                f"- {loop['type']} на строке {loop['line']}, "
                f"глубина {loop['depth']}, "
                f"переменная: {loop['var']}"
            )

        report.append("")
        report.append("Возможные проблемы и рекомендации:")

        if max_depth >= 2:
            report.append(
                "- Обнаружены вложенные циклы → возможна высокая вычислительная сложность\n"
                "  Рекомендации:\n"
                "   • рассмотреть перестановку циклов\n"
                "   • использовать развертку внутреннего цикла\n"
                "   • проверить локальность доступа к данным"
            )
        else:
            report.append(
                "- Вложенность небольшая → критичных проблем не обнаружено"
            )

        return "\n".join(report)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Loop Optimizer Lite (Level 2)")
    parser.add_argument("--file", required=True, help="Python-файл для анализа")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        code = f.read()

    optimizer = LoopOptimizerLite(code)
    report = optimizer.analyze()
    print(report)


if __name__ == "__main__":
    main()
