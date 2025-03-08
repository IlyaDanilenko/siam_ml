import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from catboost import CatBoostClassifier


def get_sample(index, frame):
    uuid = frame.iloc[index, 0]
    with open(f'data/data/data/{uuid}', 'r') as f:
        content = f.read()

    rows = [line.split('\t') for line in content.strip().split('\n')]
    data = pd.DataFrame(rows, columns=['time', 'delta_p', 'p_'], dtype=float)

    return data


# Функция для интерполяции временного ряда с использованием кубического сплайна
def interpolate_series(series, target_length):
    original_length = len(series)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)
    cs = CubicSpline(original_indices, series)
    return cs(target_indices)


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
            series = np.array(get_sample(idx, df)[['delta_p', 'p_']])
            if len(series) > 2:
                X.append(series)
                y.append(np.array(row.iloc[3:11]))
        except ValueError:
            continue

    return X, np.array(y)


# Подготовка данных
def prepare_data(X, y):
    # Интерполяция рядов
    X_interp = np.array(
        [[interpolate_series(series.T[0], TARGET_LENGTH), interpolate_series(series.T[1], TARGET_LENGTH)]
         for series in X])
    return X_interp, y


# Создание набора данных для PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X.reshape(X.shape[2] * X.shape[0], X.shape[1])).reshape(X.shape[0], X.shape[2])

        self.X = X_pca
        self.y = y


TARGET_LENGTH = 100
NUM_FEATURES = 2
NUM_CLASSES = 7
WINDOW_SIZE = 5

X, y = prepare_data(*generate_data('test'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

X_train = train_dataset.X
y_train = train_dataset.y
X_test = test_dataset.X
y_test = test_dataset.y

dict_models = {}

for j in range(NUM_CLASSES):
    model = CatBoostClassifier(
            iterations=130,
            learning_rate=0.1,
            depth=15,
            loss_function='Logloss',
            verbose=10
        )

    try:
        model.fit(X_train, y_train[:, j])
    except:
        model = lambda x: 1 if sum(y_train) > 0 else 0

    dict_models[j] = model
    break

for i in range(NUM_CLASSES):
    model = dict_models[i]

    try:
        outputs = model.predict(X_test)
    except:
        outputs = np.array([model(1)])

    print(f'Класс {i} accuracy: {accuracy_score(outputs, y_test[:, i].astype(int))}')
    print(f'Класс {i} precession: {precision_score(outputs, y_test[:, i].astype(int))}')
    print(f'Класс {i} recall: {recall_score(outputs, y_test[:, i].astype(int), zero_division=0)}')
    print(f'Класс {i} f1: {f1_score(outputs, y_test[:, i].astype(int))}')
