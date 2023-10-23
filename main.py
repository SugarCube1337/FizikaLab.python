import matplotlib.pyplot as plt

# Значения массы (в граммах)
m = [50, 100, 150, 200, 250]

# Соответствующие резонансные частоты (в Гц)
f = [15.6, 21.2, 25.3, 28.0, 35.0]

# Рассчитываем квадрат резонансной частоты
f_squared = [freq ** 2 for freq in f]

# Рассчитываем силу натяжения T = mg (в Ньютонах)
g = 9.82  # ускорение свободного падения (м/с^2)
T = [mass * g / 1000 for mass in m]  # переводим массу из граммов в килограммы

# Строим график
plt.figure(figsize=(8, 6))
plt.plot(T, f_squared, marker='o', linestyle='-', color='b')
plt.xlabel('Сила натяжения, Н')
plt.ylabel('Квадрат резонансной частоты, Гц^2')
plt.title('Зависимость квадрата резонансной частоты от силы натяжения струны')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

# Заданные данные
forces = [1178.4, 1473, 1767.6, 2062.2, 2356.8, 2651.4]
harmonics = list(range(1, 6))

# Резонансные частоты для каждой гармоники и силы натяжения
resonant_frequencies = [
    [5.4, 11.4, 17.5, 23.4, 28.2],
    [6, 13.1, 18.2, 23.9, 31.9],
    [7.1, 14.2, 20.5, 27.4, 34],
    [7.8, 15.1, 22.5, 29, 40],
    [8.4, 16.7, 23.8, 34.3, 42.9],
    [8.9, 17.6, 25.8, 35.7, 45.7]
]

# Построение графиков
for i, force in enumerate(forces):
    plt.plot(harmonics, resonant_frequencies[i], marker='o', label=f'T = {force} Н')

plt.xlabel('Номер гармоники')
plt.ylabel('Резонансные частоты, Гц')
plt.title('Зависимость резонансных частот от номера гармоники')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Заданные данные
forces = np.array([1178.4, 1473, 1767.6, 2062.2, 2356.8, 2651.4])
harmonics = np.array([1, 2, 3, 4, 5, 6])
length = 1.45  # Длина струны в метрах

# Линейная аппроксимация: y = mx + b
def linear_fit(x, m, b):
    return m * x + b

# Расчет u^2 как функции силы натяжения T
# Проведение линейной аппроксимации и расчет углового коэффициента m
m, _ = curve_fit(linear_fit, 2 * np.pi * harmonics * length, forces)

# Построение графика
plt.scatter(2 * np.pi * harmonics * length, forces, label='Экспериментальные данные')
plt.plot(2 * np.pi * harmonics * length, linear_fit(2 * np.pi * harmonics * length, m[0], m[1]), color='red', label='Линейная аппроксимация')
plt.xlabel('2πknℓ')
plt.ylabel('Сила натяжения T')
plt.title('Линейная аппроксимация зависимости 2πknℓ от T')
plt.legend()
plt.grid(True)
plt.show()

# Вывод результатов
print(f"Угловой коэффициент линейной аппроксимации m: {m[0]:.2f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Заданные данные (данные из таблицы 2)
masses = np.array([120, 150, 180, 210, 240, 270]) * 1e-3  # в кг
forces = np.array([1178.4, 1473, 1767.6, 2062.2, 2356.8, 2651.4])  # в Н
harmonics_squared = np.array([1, 4, 9, 16, 25, 36])  # k_n^2
length = 1.45  # длина струны в м

# Расчет u^2 (k_n^2 / (4 * pi^2) * T / (rho_l * l))
u_squared = (harmonics_squared / (4 * np.pi**2)) * (forces / (masses * length))

# Построение графика
plt.scatter(forces, u_squared, label='Экспериментальные данные')
plt.xlabel('Сила натяжения T (Н)')
plt.ylabel('Квадрат фазовой скорости u^2 (м^2/с^2)')
plt.title('Зависимость квадрата фазовой скорости от силы натяжения')
plt.legend()
plt.grid(True)
plt.show()

# Линейная аппроксимация: y = mx + b
def linear_fit(x, m, b):
    return m * x + b

# Аппроксимация данных методом наименьших квадратов
m, _ = curve_fit(linear_fit, forces, u_squared)

# Угловой коэффициент графика
angular_coefficient = m[0]

# Линейная плотность струны rho_l (rho_l = k / (4 * pi^2 * m))
linear_density = angular_coefficient / (4 * np.pi**2)

# Погрешность углового коэффициента
angular_coefficient_error = np.sqrt(np.diag(_)[0])

# Погрешность линейной плотности
linear_density_error = angular_coefficient_error / (4 * np.pi**2)

# Вывод результатов
print(f"Угловой коэффициент графика: {angular_coefficient:.5f} (м^2/с^2/Н)")
print(f"Линейная плотность струны rho_l: {linear_density:.5f} кг/м")
print(f"Погрешность углового коэффициента: {angular_coefficient_error:.5f} (м^2/с^2/Н)")
print(f"Погрешность линейной плотности: {linear_density_error:.5f} кг/м")