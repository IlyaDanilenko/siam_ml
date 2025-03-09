import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_sample(index, frame):
    uuid = frame.iloc[index, 0]
    with open(f'data/data/data/{uuid}', 'r') as f:
        content = f.read()

    rows = [line.split('\t') for line in content.strip().split('\n')]
    data = pd.DataFrame(rows, columns=['time', 'delta_p', 'p_'], dtype=float)
    return data


def generate_data(mode):
    X = []
    y = []
    if mode == 'train':
        df = pd.read_csv('data/markup_train.csv')
        df = df[0:10000]
    else:
        df = pd.read_csv('data/hq_markup_train.csv')

    for idx, row in df.iterrows():
        try:
            series = np.array(get_sample(idx, df))
            if len(series) > 2:
                X.append(series)
                y.append(np.array(row.iloc[3:15]))
        except ValueError:
            continue

    return X, np.array(y)


def fft(sig, fs):
    T = 1 / fs
    n = len(sig)  # Длина сигнала
    fft_result = np.fft.fft(sig)  # БПФ
    amplitudes = np.abs(fft_result)  # Амплитуды

    # Частотная ось
    frequencies = np.fft.fftfreq(n, T)  # Массив частот

    half_n = n // 2
    frequencies = frequencies[:half_n]
    amplitudes = amplitudes[:half_n]
    return frequencies, amplitudes


def apply_fft(t, signal, signal_2, fs):
    frequencies, amplitudes = fft(signal, fs)
    frequencies_2, amplitudes_2 = fft(signal_2, fs)

    plt.figure(figsize=(12, 6))

    # Временной ряд
    plt.subplot(2, 1, 1)
    plt.plot(t, signal_2)
    plt.title('Временной ряд')
    plt.xlabel('Время (с)')
    plt.ylabel('Производная')

    # Частотный спектр
    plt.subplot(2, 1, 2)
    plt.plot(frequencies_2, amplitudes_2)
    plt.title('Частотный спектр (БПФ)')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(0, 200)  # Ограничим диапазон для наглядности

    plt.tight_layout()
    plt.show()


X, y = generate_data('test')
for i in range(len(X)):
    apply_fft(X[i].T[0], X[i].T[1], X[i].T[2], 1000)
    print(y[i])
