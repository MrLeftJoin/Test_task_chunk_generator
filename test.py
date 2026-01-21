import pandas as pd
import numpy as np
import unittest
from memory_profiler import profile


def get_chunks(df: pd.DataFrame, column: str, min_size: int):
    if df.empty:
        return
    # ищем длину фрейма
    n = len(df)
    # сравниваем исходную колонку со сдвинутой (lead 1). Ставим True если серия одинаковых дат прервалась
    marks = df[column].ne(df[column].shift())
    # через numpy ищем индексы для каждой прерванной серии (проще говоря - старта новой серии)
    change_indices = np.where(marks)[0].tolist()
    # Добавляем конец для последнего среза
    change_indices.append(n)

    start_pos = 0
    for i in range(1, len(change_indices)):
        current_pos = change_indices[i]
        # Проверяем накопленный размер
        if (current_pos - start_pos) >= min_size or current_pos == n:
            # отдаем по чанку гененратором
            yield df.iloc[start_pos:current_pos]
            # двигаем и запоминаем последнюю позицию
            start_pos = current_pos


# бизнес тесты
class TestChunking(unittest.TestCase):
    def setUp(self):
        # Набор данных из примера
        # 2023-01-01 00:00:01 -> 2 строки
        # 2023-01-01 00:00:02 -> 3 строки
        # 2023-01-01 00:00:03 -> 1 строка
        # Итого: 6 строк
        dates = (["2023-01-01 00:00:01"] * 2 +
                 ["2023-01-01 00:00:02"] * 3 +
                 ["2023-01-01 00:00:03"] * 1)
        self.df = pd.DataFrame({"dt": dates})

    def test_case_size_1_to_2(self):
        # Кейс: размер между 1 и 2 -> результат 3 чанка (2, 3, 1)
        for size in [1, 2]:
            with self.subTest(min_size=size):
                chunks = list(get_chunks(self.df, 'dt', size))
                # проверили количество чанков
                self.assertEqual(len(chunks), 3)
                # сверили 2 списка с длинами чанков
                self.assertEqual([len(c) for c in chunks], [2, 3, 1])

    def test_case_size_3_to_5(self):
        # Кейс: размер между 3 и 5 -> результат 2 чанка (5, 1)
        for size in [3, 4, 5]:
            with self.subTest(min_size=size):
                chunks = list(get_chunks(self.df, 'dt', size))
                self.assertEqual(len(chunks), 2)
                self.assertEqual([len(c) for c in chunks], [5, 1])

    def test_case_size_6_plus(self):
        # Кейс: размер от 6 и выше -> весь фрейм (1 чанк)
        for size in [6, 10, 100]:
            with self.subTest(min_size=size):
                chunks = list(get_chunks(self.df, 'dt', size))
                self.assertEqual(len(chunks), 1)
                self.assertEqual(len(chunks[0]), 6)

    def test_no_overlap_integrity(self):
        # Условие: даты не должны пересекаться между чанками
        chunks = list(get_chunks(self.df, 'dt', 2))
        for i in range(len(chunks) - 1):
            last_date = chunks[i]['dt'].iloc[-1]
            next_first_date = chunks[i+1]['dt'].iloc[0]
            self.assertNotEqual(last_date, next_first_date)


# тесты производительности
@profile
def run_test():
    # Создаем тяжелый DF
    print("Создаем данные...")
    # 100 групп по 100к строк
    data = {
        'dt': np.repeat(np.arange(100), 100000),
        'value': np.random.randn(10000000)
    }
    df = pd.DataFrame(data)

    print("Начинаем итерацию по чанкам...")
    # Итерируемся по чанкам
    # Если бы мы копировали данные, память бы росла на каждой итерации
    for chunk in get_chunks(df, 'dt', 500000):
        # Прогреваем вычисления, чтобы пандас не "ленился"
        _ = chunk['value'].sum()

    print("Готово.")


if __name__ == '__main__':
    # Запуск тестов
    unittest.main(argv=[''], exit=False)
    run_test()
