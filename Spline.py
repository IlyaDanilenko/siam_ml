import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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
    X_interp = np.array([[interpolate_series(series.T[0], TARGET_LENGTH), interpolate_series(series.T[1], TARGET_LENGTH)]
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

    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_interp.reshape(-1, 1)).reshape(-1, TARGET_LENGTH, NUM_FEATURES)
    return X_scaled, y


# Создание набора данных для PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.astype('int32'), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeSeriesModel(nn.Module):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()

        # CNN ветка
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # LSTM ветка
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=4, batch_first=True, bidirectional=True),
        )

        # Объединение ветвей
        self.fc_combined = nn.Sequential(
                nn.Linear(11776, 8),
                nn.Sigmoid()
        )

    def forward(self, x):
        # lstm
        lstm_out, _ = self.lstm(x)

        # CNN ветка
        cnn_out = self.conv(lstm_out.permute(0, 2, 1))  # Изменяем размерность для Conv1D

        # Объединение ветвей
        combined = torch.relu(self.fc_combined(cnn_out))

        return combined


# Параметры
TARGET_LENGTH = 100  # Целевая длина временного ряда
NUM_FEATURES = 2  # Количество признаков в временном ряде (1 для одномерного ряда)
NUM_CLASSES = 8  # Количество классификационных параметров

X, y = prepare_data(*generate_data('test'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Создание датасетов и загрузчиков
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Создание модели
model = TimeSeriesModel()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

# Оценка модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += targets.size(0) * NUM_CLASSES
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
