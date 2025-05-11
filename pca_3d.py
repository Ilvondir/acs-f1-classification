from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42


datasets = Path('datasets')
data_labels = {}
for i, dir in enumerate(datasets.iterdir()):
    data_labels[dir.name] = i


colnames = ['phAngle', 'freq', 'reacPower', 'power', 'rmsVolt', 'rmsCur']

data = []
labels = []

for dir in datasets.iterdir():
    label = datasets / dir.name

    for file in label.iterdir():
        
        with open(file, 'r') as f:
            file_content = f.readlines()
        
        file_content = [line[1:-2] for line in file_content if line[0] != '#']
        columns = np.array([list(map(float, row.split(' '))) for row in file_content])
        padded = tf.keras.preprocessing.sequence.pad_sequences(columns, maxlen=360, dtype='float32', padding='pre').T

        data.append(padded)
        labels.append(data_labels[file.parent.name])

data, labels = np.array(data), np.array(labels)

class Reshape3DTo2D(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(-1, 6)
    

class Reshape2DTo3D(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(-1, 360, 6)


preprocess = Pipeline([
    ('to2D', Reshape3DTo2D()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('to3D', Reshape2DTo3D())
])



pca = PCA(n_components=3, random_state=SEED)
data_scaled = preprocess.fit_transform(data)
base_shape = data_scaled.shape
data_scaled = data_scaled.reshape((base_shape[0], base_shape[1]*base_shape[2]))
data_pca = pca.fit_transform(data_scaled)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels, alpha=0.8)
plt.show()

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))