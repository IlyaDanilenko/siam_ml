from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error


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
    maes = []

    for degree in range(1, 15):
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )

        model.fit(train[t][:, 0].reshape(-1, 1), train[t][:, 1])
        y_pred = model.predict(train[t][:, 0].reshape(-1, 1))

        maes.append(mean_absolute_error(y_pred, train[t][:, 1]))

    smallest_mae = min(maes)
    idx = maes.index(smallest_mae)

    model = make_pipeline(
        PolynomialFeatures(idx, include_bias=False),
        LinearRegression()
    )

    model.fit(train[t][:, 0].reshape(-1, 1), train[t][:, 1])

    m = []
    # все коэффициенты
    for coef in model.named_steps['linearregression'].coef_:
        m.append(mean_absolute_error([y[t]], [coef]))
    # свободный член
    m.append(model.named_steps['linearregression'].intercept_)

    mae.append(min(m))

print(sum(mae) / len(mae))




