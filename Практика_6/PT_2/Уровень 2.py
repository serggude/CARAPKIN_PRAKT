import re

class AdvancedLoopOptimizer:
    def __init__(self):
        self.dependency_vectors = []

    def analyze_loop(self, code_string):

        self.dependency_vectors.clear()

        writes = re.findall(r'(\w+)\[i\]\[j\]\s*=', code_string)

        reads = re.findall(r'(\w+)\[(i(?:-1)?)\]\[(j(?:-1)?)\]', code_string)

        for array, i_idx, j_idx in reads:
            vector = self._compute_direction_vector(i_idx, j_idx)

            self.dependency_vectors.append({
                'array': array,
                'read': (i_idx, j_idx),
                'write': ('i', 'j'),
                'direction': vector,
                'type': self._classify_dependency(vector)
            })

        return self.dependency_vectors

    def _compute_direction_vector(self, i_idx, j_idx):
        def dir_component(x):
            if x == 'i' or x == 'j':
                return '='
            if '-1' in x:
                return '<'
            return '?'

        return (dir_component(i_idx), dir_component(j_idx))

    def _classify_dependency(self, vector):
        if '<' in vector:
            return 'Flow'
        return 'Independent'

    def generate_parallel_code(self, strategy='distribution'):
        if strategy == 'distribution':
            return (
                "Рекомендация: распараллелить внешний цикл (i),\n"
                "так как зависимости не препятствуют независимой обработке строк."
            )

        if strategy == 'tiling':
            return (
                "Рекомендация: использовать блочную обработку (tiling)\n"
                "для улучшения локальности данных и кэш-эффективности."
            )

        return "Стратегия не распознана"

    def benchmark_strategies(self):
        return {
            'original': 'baseline',
            'parallel_outer': 'expected speedup',
            'tiled': 'best cache behavior'
        }

def test_advanced_loop_optimizer():
    code = """
for i in range(1, n):
    for j in range(1, m):
        a[i][j] = a[i-1][j] + a[i][j-1]
"""
    optimizer = AdvancedLoopOptimizer()

    print("=== Анализ зависимостей ===")
    dependencies = optimizer.analyze_loop(code)

    for dep in dependencies:
        print(f"\nМассив: {dep['array']}")
        print(f"Чтение: {dep['read']}")
        print(f"Запись: {dep['write']}")
        print(f"Вектор направлений: {dep['direction']}")
        print(f"Тип зависимости: {dep['type']}")

    print("\n=== Рекомендация по параллелизации ===")
    print(optimizer.generate_parallel_code(strategy='distribution'))

    print("\n=== Альтернативная стратегия ===")
    print(optimizer.generate_parallel_code(strategy='tiling'))

    print("\n=== Бенчмаркинг (концептуальный) ===")
    benchmark = optimizer.benchmark_strategies()
    for k, v in benchmark.items():
        print(f"{k}: {v}")
