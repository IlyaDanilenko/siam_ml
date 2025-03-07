import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def get_sample(index, frame):
    uuid = frame.iloc[index, 0]
    with open(f'data/data/data/{uuid}', 'r') as f:
        content = f.read()

    rows = [line.split('\t') for line in content.strip().split('\n')]
    data = pd.DataFrame(rows, columns=['time', 'delta_p', 'p_'], dtype=float)

    pca = PCA(n_components=1)
    data['pca'] = pca.fit_transform(data[['delta_p', 'p_']])
    data.to_csv(f'data/csv/{uuid}.csv')

    return data


df = pd.read_csv('data/hq_markup_train.csv')
series = []

for idx, row in df.iterrows():
    try:
        series.append(np.array(get_sample(idx, df)))
    except ValueError:
        continue
