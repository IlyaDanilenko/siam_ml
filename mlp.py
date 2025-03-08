import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


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

    for i in range(0):
        plt.figure(figsize=(8, 6))

        # Ряд 1: точки (формат 'o' означает круглые маркеры)
        plt.plot(X_interp[i][0], 'o', label='delta', color='blue')
        plt.plot(X_interp[i][1], 's', label='diff', color='green')

        plt.plot(X[i].T[0], '-', label='aprox_delta', color='red')
        plt.plot(X[i].T[1], '-', label='aprox_diff', color='black')

        plt.legend()
        plt.title('График с точками и линией')
        plt.xlabel('X-ось')
        plt.ylabel('Y-ось')

        # Отображение графика
        plt.grid(True)  # Добавляем сетку для удобства восприятия
        plt.show()
    return X_interp, y


# Создание набора данных для PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X.reshape(X.shape[2] * X.shape[0], X.shape[1])).reshape(X.shape[0], X.shape[2])

        self.X = torch.tensor(X_pca, dtype=torch.float32)
        self.y = torch.tensor(y.astype(int), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_combined = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        combined = self.fc_combined(x)
        return combined


TARGET_LENGTH = 100
NUM_FEATURES = 2
NUM_CLASSES = 7
WINDOW_SIZE = 5

X, y = prepare_data(*generate_data('test'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loaders = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loaders = DataLoader(test_dataset, batch_size=1, shuffle=True)

dict_models = {}
for j in range(NUM_CLASSES):
    list_models = []
    for i in range(0, TARGET_LENGTH, WINDOW_SIZE):
        model = MLP()
        criterion = nn.BCELoss()
        optimizer = optim.NAdam(model.parameters(), lr=0.0001)

        for epoch in range(50):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loaders:
                optimizer.zero_grad()
                outputs = model(inputs[:, i:i+5])

                loss = criterion(outputs, targets[:, j].reshape(-1, 1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loaders):.4f}')

        list_models.append([model, i])
    dict_models[j] = list_models

for i in range(NUM_CLASSES):
    list_models = dict_models[i]
    preds = []
    targs = []

    for inputs, targets in test_loaders:
        target = int(targets[:, i])
        predicted = []
        for model, idx in list_models:
            model.eval()

            with torch.no_grad():
                outputs = model(inputs[:, idx:idx+5])
                predicted.append(1 if float(outputs) > 0.5 else 0)

        answer = int(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predicted))

        preds.append(answer)
        targs.append(target)

    print(f'Класс {i} accuracy: {accuracy_score(preds, targs)}')
    print(f'Класс {i} precession: {precision_score(preds, targs)}')
    print(f'Класс {i} recall: {recall_score(preds, targs)}')
    print(f'Класс {i} f1: {f1_score(preds, targs)}')

