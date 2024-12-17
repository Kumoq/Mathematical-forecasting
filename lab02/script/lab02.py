import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


class LinearModelAnalysis:
    def __init__(self):
        self.data = None
        self.target = None
        self.features = None
        self.model = None

    def read_data(self, file_path, file_format='excel'):
        """
        Чтение данных из файла в зависимости от формата.
        """
        try:
            if file_format == 'excel':
                self.data = pd.read_excel(file_path)
            elif file_format == 'csv':
                self.data = pd.read_csv(file_path)
            else:
                print(f"Не поддерживаемый формат файла: {file_format}")
                return None
            print("Данные успешно загружены.")
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")

    def select_target(self):
        """Позволяет пользователю выбрать целевую переменную."""
        if self.data is not None:
            print("Доступные столбцы для выбора таргета:")
            print(self.data.columns)
            target_column = input("Введите название столбца для таргета: ").strip()

            if target_column in self.data.columns:
                self.target = target_column
                self.features = self.data.drop(columns=[self.target])  # Все остальные столбцы становятся фичами
                print(f"Целевая переменная '{self.target}' успешно выбрана.")
                print("Доступные признаки (фичи):")
                print(self.features.columns)
            else:
                print(f"Ошибка: столбец '{target_column}' не найден в данных.")
        else:
            print("Данные не загружены. Пожалуйста, загрузите данные сначала.")

    def significance_testing(self):
        """Проводим статистическую проверку значимости факторов с использованием OLS."""
        if self.data is None or self.target is None:
            print("Данные или целевая переменная не выбраны.")
            return

        # Добавление константы для модели
        X = sm.add_constant(self.features)
        y = self.data[self.target]

        # Строим модель
        model = sm.OLS(y, X).fit()

        # Выводим отчет о модели
        print("\nОтчет о модели:")
        print(model.summary())

        # Запрашиваем уровень значимости
        try:
            significance_level = float(input("\nВведите уровень значимости для отсевов (например, 0.05): ").strip())
        except ValueError:
            print("Неверный ввод. Уровень значимости должен быть числом.")
            return

        # Проверка значимости факторов
        p_values = model.pvalues
        significant_factors = p_values[p_values <= significance_level]

        if 'const' in significant_factors.index:
            significant_factors = significant_factors.drop('const')

        if len(significant_factors) > 0:
            print(f"\nФакторы с p-значениями <= {significance_level}:")
            print(significant_factors)
            # Обновляем список признаков, оставляя только значимые факторы
            self.features = self.features[significant_factors.index]
            print('\nПризнаки успешно обновлены')
        else:
            print(f"\nНет факторов с p-значениями <= {significance_level}.")

        choice = input("\nХотите повторить отсев по уровню значимости (y/n)? ").strip().lower()
        if choice == 'y':
            self.significance_testing()
        else:
            print("Завершаем проверку значимости.")

    def correlation_filter(self):
        """Отбор признаков по коэффициенту корреляции (между собой)."""
        if self.data is None or self.target is None:
            print("Данные или целевая переменная не выбраны.")
            return

        # Рассчитываем корреляцию между признаками
        correlation_matrix = self.features.corr()

        # Запрашиваем порог корреляции
        try:
            correlation_threshold = float(input("\nВведите порог корреляции для отсевов (например, 0.9): ").strip())
        except ValueError:
            print("Неверный ввод. Порог корреляции должен быть числом.")
            return

        # Выводим корреляцию между признаками
        print("\nКорреляционная матрица:")
        print(correlation_matrix)

        # Отбираем признаки, которые имеют корреляцию ниже порога
        to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    to_drop.add(colname)

        if len(to_drop) > 0:
            print(f"\nПризнаки, которые будут удалены из-за высокой корреляции: {to_drop}")
            # Обновляем список признаков, исключая сильно коррелирующие
            self.features = self.features.drop(columns=to_drop)
            print("\nПризнаки успешно обновлены.")
        else:
            print("\nНет признаков, которые превышают заданный порог корреляции.")

        choice = input("\nХотите повторить Отбор признаков по коэффициенту корреляции (между собой) (y/n)?").strip().lower()
        if choice == 'y':
            self.significance_testing()
        else:
            print("Завершаем проверку значимости.")

    def correlation_with_target(self):
        """Отбор признаков по коэффициенту корреляции с откликом."""
        if self.data is None or self.target is None:
            print("Данные или целевая переменная не выбраны.")
            return

        # Рассчитываем корреляцию между признаками и целевой переменной
        correlation_with_target = self.features.corrwith(self.data[self.target])

        # Запрашиваем порог корреляции
        try:
            correlation_threshold = float(input("\nВведите порог корреляции для отсевов (например, 0.5): ").strip())
        except ValueError:
            print("Неверный ввод. Порог корреляции должен быть числом.")
            return

        # Выводим корреляцию между признаками и откликом
        print("\nКорреляция признаков с целевой переменной:")
        print(correlation_with_target)

        # Отбираем признаки, которые имеют корреляцию ниже порога
        to_drop = correlation_with_target[abs(correlation_with_target) < correlation_threshold].index

        if len(to_drop) > 0:
            print(f"\nПризнаки, которые будут удалены из-за низкой корреляции с откликом: {to_drop}")
            # Обновляем список признаков, исключая те, которые имеют низкую корреляцию с откликом
            self.features = self.features.drop(columns=to_drop)
            print("\nПризнаки успешно обновлены.")
        else:
            print("\nВсе признаки имеют достаточную корреляцию с откликом.")

        choice = input(
            "\nХотите повторить Отбор признаков по коэффициенту корреляции с откликом (y/n)?").strip().lower()
        if choice == 'y':
            self.correlation_with_target()
        else:
            print("Завершаем проверку корреляции.")

    def train_model(self):
        try:
            """Обучение модели с разделением на обучающую и тестовую выборки."""
            if self.data is None or self.target is None:
                print("Данные или целевая переменная не выбраны.")
                return

            # Разделение данных на обучающую и тестовую выборки
            X = self.features
            y = self.data[self.target]

            # Разделение на обучающие и тестовые данные (80%/20%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Добавление константы для модели
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)

            # Обучаем модель
            self.model = sm.OLS(y_train, X_train_const).fit()

            # Предсказания на тестовой выборке
            y_pred = self.model.predict(X_test_const)

            # Коэффициент детерминации (R²)
            r_squared = self.model.rsquared
            print(f"\nКоэффициент детерминации R²: {r_squared}")

            # F-статистика для проверки значимости модели
            f_statistic = self.model.fvalue
            print(f"F-статистика модели: {f_statistic}")

            # Оценка адекватности модели
            if f_statistic > 0:
                print("Модель адекватна.")
            else:
                print("Модель неадекватна.")

            # Ошибка (E) - Средняя ошибка по формуле
            error = np.mean(np.abs((y_test - y_pred) / y_test))
            print(f"Ошибка (E): {error}")

            # RMSE - Среднеквадратическая ошибка
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            print(f"RMSE: {rmse}")

        except Exception as e:
            print(f"Произошла ошибка в процессе обучения модели: {e}")

    def predict_from_input(self):
        """Выполнить предсказание с использованием значений, введенных пользователем."""
        if self.model is None:
            print("Модель не обучена. Сначала обучите модель.")
            return

        # Запрос значений для признаков у пользователя
        input_data = {}
        for feature in self.features.columns:
            value = input(f"Введите значение для признака '{feature}': ").strip()
            try:
                input_data[feature] = float(value)
            except ValueError:
                print(f"Неверный ввод для признака '{feature}', должно быть числовое значение.")
                return

        # Преобразуем введенные данные в DataFrame
        input_df = pd.DataFrame([input_data])

        # Добавление константы для предсказания
        input_df_const = sm.add_constant(input_df, has_constant='add')

        # Предсказание
        prediction = self.model.predict(input_df_const)
        print(f"Предсказанное значение для целевой переменной: {prediction[0]}")


def show_statistics_menu():
    print("\nВарианты отбора:"
          "\n1. По статистической значимости"
          "\n2. По коэффициенту корреляции - связь между факторами (между собой)"
          "\n3. По коэффициенту корреляции - связь между факторами и откликом"
          "\n4. Назад ")


def show_menu():
    print("\nФункционал программы:"
          "\n 1. Загрузка данных"
          "\n 2. Выбор таргета"
          "\n 3. Отбор значимых факторов"
          "\n 4. Обучить модель"
          "\n 5. Выполнить предсказание"
          "\n 6. Выход")

if __name__ == "__main__":

    linear_model_analysis = LinearModelAnalysis()

    while True:
        show_menu()
        try:
            choice = int(input("\nВведите номер опции: "))

            if choice == 1:
                # Запрос пути к файлу и формата
                file_path = input("Введите путь к файлу: ")
                # Используем excel как формат по умолчанию
                file_format = input(
                    "Введите формат файла (excel/csv) или нажмите Enter для использования excel по умолчанию: ").strip().lower()

                # Если формат не введен, используем excel
                if not file_format:
                    file_format = 'excel'

                linear_model_analysis.read_data(file_path, file_format)


            elif choice == 2:
                linear_model_analysis.select_target()

            elif choice == 3:

                show_statistics_menu()

                while True:
                    try:
                        choice = int(input("\nВведите номер опции: "))
                        if choice == 1:
                            linear_model_analysis.significance_testing()

                            show_statistics_menu()

                        elif choice == 2:
                            linear_model_analysis.correlation_filter()

                            show_statistics_menu()

                        elif choice == 3:
                            linear_model_analysis.correlation_with_target()
                            show_statistics_menu()
                        elif choice == 4:
                            break

                    except ValueError:
                        print("Неверный ввод. Пожалуйста, введите номер опции.")

            elif choice == 4:
                linear_model_analysis.train_model()

            elif choice == 5:
                linear_model_analysis.predict_from_input()

            elif choice == 5:
                print("Завершаем программу.")
                break

            else:
                print("Неверный выбор. Пожалуйста, выберите одну из доступных опций.")

        except ValueError:
            print("Неверный ввод. Пожалуйста, введите номер опции.")