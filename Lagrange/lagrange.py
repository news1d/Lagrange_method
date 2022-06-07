'''
1. Интерполяция полиномом Лагранжа. Оценить точность интерполяции для тестовой функции, заданной аналитически в зависимости от количества точек (10 – 50 точек. Для тестовой задачи можно использовать точки, вычисленные для какой-либо типовой функции).
2. Построить точечную диаграмму тестовой функции на заданном интервале с заданием погреш-ностей по Х и Y.
3. Запись исходных данных и результатов расчётов.
4. Запись в бинарный файл всех пользовательских настроек и считывание их оттуда.
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

def func(x):
    return np.exp(x)

def lagrange_poly(x, Y, a, b):
    L = Y.copy()
    x = (x - a) * (len(Y) - 1) / (b - a)
    for i in range(len(Y)):
        for j in range(len(Y)):
            if i != j:
                L[i] *= (x-j) / (i - j)
    return np.sum(L)

try:
    with open('settings.pkl', 'rb') as f:
        config = pickle.load(f)
        a = ''
        while a.lower() not in ['да', 'нет', 'д', 'н']:
            print("Файл настроек найден. Вы хотите загрузить его (Да / Нет)?")
            a = input()
            if a in ['нет', 'н']:
                raise FileNotFoundError
except FileNotFoundError as e:
    # Файл не найден. Конфиг по умолчанию
    config = {
        'K': [10,20,30,40,50],
        'n': 200,
        'a': -np.pi*8,
        'b': np.pi*8
    }
    
    a = ''
    while a.lower() not in ['да', 'нет', 'д', 'н']:
        print('Файл с настройками не найден. Загружены настройки по умолчанию.')
        print(f'config: {config}')
        print('Вы хотите сохранить настройки по умолчанию (Да / Нет)?')
        a = input()
        if a in ['да', 'д']:
            with open('settings.pkl', 'wb') as f:
                pickle.dump(config, f)

print(f'config: {config}')
K = config['K']
n = config['n']
a = config['a']
b = config['b']
Dif = []

for k in K:
    X = np.linspace(a, b, n)
    Y = func(X)

    ptsX = np.linspace(a, b, k)
    ptsY = func(ptsX)
    L = np.array([lagrange_poly(x, ptsY, a, b) for x in X])
    
    # Save integral of difference function
    d = np.abs(L-Y).mean() * (b-a)
    Dif.append(d)
    
    # Plot
    plt.title(f"Dif: {d}")
    plt.plot(X, Y, label="exp(x)")
    plt.plot(X, L, label=f"lagrange_{k}")
    plt.scatter(ptsX, ptsY)

    plt.legend()
    plt.savefig(f'lagrange_{k}.png')
    plt.show()
    print(f'График сохранен lagrange_{k}.png')

results = []
results.append(f'Результат для k={k} точек:')
results.append('X:')
results.append(str(X))
results.append('Y:')
results.append(str(Y))
results.append('L:')
results.append(str(L))
results.append(f'Интеграл от абсолютной разности (L-Y): {Dif[-1]}')

print('3. Результат вычислений')
print('\n'.join(results))
with open('results_log.txt', 'w') as f:
    f.write('\n'.join(results))
    print('Результат записан в results_log.txt файл')

plt.xlabel("Количество точек")
plt.ylabel("Интеграл модуля разности")
plt.plot(K, Dif)
plt.savefig('difference_from_k.png')
plt.show()
print(f'График сохранен difference_from_k.png')
