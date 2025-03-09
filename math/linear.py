from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor


def get_sample(index, frame):
    uuid = frame.iloc[index, 0]
    with open(f'data/data/data/{uuid}', 'r') as f:
        content = f.read()

    rows = [line.split('\t') for line in content.strip().split('\n')]
    data = pd.DataFrame(rows, columns=['time', 'delta_p', 'p_'], dtype=float)

    data['p_'] = np.log(data['p_'])
    data['time'] = np.log(data['time'])

    return data


# # Пример генерации синтетических данных
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
            series = np.array(get_sample(idx, df)[['time', 'p_']])
            if len(series) > 2:
                X.append(series)
                y.append(np.array(row.iloc[11:]))
        except ValueError:
            continue

    return X, np.array(y)


X, targets = generate_data('test')

target = targets[:, 0]
not_nan = []
for t in range(len(target)):
    if not math.isnan(target[t]):
        not_nan.append(t)

y = []
for idx in not_nan:
    y.append(target[idx])

train = []
for idx in not_nan:
    train.append(X[idx])

mae = []
for t in range(len(train)):
    degree = 2
    # model = make_pipeline(
    #     PolynomialFeatures(degree, include_bias=True),
    #     CatBoostRegressor(learning_rate=0.1)
    # )
    # model.fit(train[t][:, 0].reshape(-1, 1), train[t][:, 1])

    # coef = model.named_steps['linearregression'].intercept_
    coefficients = np.polyfit(train[t][:, 0], train[t][:, 1], degree)

    # Создание полиномиальной функции
    poly = np.poly1d(coefficients)

    derivative = np.polyder(poly)
    x_fine = np.linspace(train[t][:, 0].min(), train[t][:, 1].max(), 1000)
    dy_dx = derivative(x_fine)
    max_growth_idx = np.argmax(dy_dx)
    max_growth = x_fine[max_growth_idx]

    delta_y = np.diff(poly(x_fine))
    max_interval_idx = np.argmax(delta_y)
    max_interval_start = x_fine[:-1][max_interval_idx]
    max_interval_end = x_fine[1:][max_interval_idx]

    new_x = []
    new_y = []
    for i in range(len(train[t][:, 0])):
        if train[t][:, 0][i] < max_interval_end:
            new_x.append(train[t][:, 0][i])
            new_y.append(train[t][:, 1][i])

    try:
        coefficients = np.polyfit(new_x, new_y, 1)

        # Создание полиномиальной функции
        poly = np.poly1d(coefficients)

        mae.append(mean_absolute_error([coefficients[-1]], [y[t]]))
    except TypeError:
        continue

print(sum(mae) / len(mae))




