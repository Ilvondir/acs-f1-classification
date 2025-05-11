# %%
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 777

# %%
datasets = Path('datasets')

data_labels = {}

for i, dir in enumerate(datasets.iterdir()):
    data_labels[dir.name] = i

data_labels

# %%
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

data.shape, labels.shape

# %%
from sklearn.base import BaseEstimator, TransformerMixin


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

# %%
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


preprocess = Pipeline([
    ('to2D', Reshape3DTo2D()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('to3D', Reshape2DTo3D())
])

preprocess

# %%
from sklearn.model_selection import train_test_split


X_train, X_validate, Y_train, Y_validate = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=SEED)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_validate, Y_validate, test_size=0.33, stratify=Y_validate, random_state=SEED)

X_train.shape

# %%
np.random.seed(SEED)

# Augmentation new data with noise
X_train_aug = []
Y_train_aug = []

for i, arr in enumerate(X_train):
    for _ in range(2):
        X_train_aug.append(arr + np.random.normal(0, 0.001, size=arr.shape))
        Y_train_aug.append(Y_train[i])

X_train_aug = np.array(X_train_aug)
Y_train_aug = np.array(Y_train_aug)

X_train = np.concatenate([X_train, X_train_aug], axis=0)
Y_train = np.concatenate([Y_train, Y_train_aug], axis=0)

X_train.shape

# %%
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(360, 6)),
        tf.keras.layers.GRU(units=256, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])

model = make_model()
model.load_weights('started.weights.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(),
              metrics=['accuracy']
)

mc1 = tf.keras.callbacks.ModelCheckpoint('best_model_1.keras', monitor='val_accuracy', mode='max', save_best_only=True)
mc2 = tf.keras.callbacks.ModelCheckpoint('best_model_2.keras', monitor='val_loss', mode='min', save_best_only=True)

preprocess.fit(X_train)
history = model.fit(
    preprocess.transform(X_train),
    tf.keras.utils.to_categorical(Y_train),
    validation_data=(preprocess.transform(X_validate), tf.keras.utils.to_categorical(Y_validate)),
    batch_size=8,
    epochs=100,
    callbacks=[mc1, mc2],
)

# %%
plt.plot(history.epoch, history.history['accuracy'], label='Train accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.title('Accuracy plot')
plt.show()

plt.plot(history.epoch, history.history['loss'], label='Train loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Loss plot')
plt.show()

# %%
model = tf.keras.models.load_model('best_model_1.keras')

preds = np.argmax(model.predict(preprocess.transform(X_test), verbose=0), axis=1)

results = pd.DataFrame({
    'preds': preds,
    'true': Y_test
})

print(f'Test accuracy: {accuracy_score(results['true'], results['preds'])}')
print(f'Test recall: {recall_score(results['true'], results['preds'], average='weighted')}')
print(f'Test precision: {precision_score(results['true'], results['preds'], average='weighted')}')
print(f'Test f1: {f1_score(results['true'], results['preds'], average='weighted')}')

cm = confusion_matrix(results['true'], results['preds'])
ConfusionMatrixDisplay(cm).plot()

# %%
preds = np.argmax(model.predict(preprocess.transform(data), verbose=0), axis=1)

results = pd.DataFrame({
    'preds': preds,
    'true': labels
})

print(f'Full dataset accuracy: {accuracy_score(results['true'], results['preds'])}')
print(f'Full dataset recall: {recall_score(results['true'], results['preds'], average='weighted')}')
print(f'Full dataset precision: {precision_score(results['true'], results['preds'], average='weighted')}')
print(f'Full dataset f1: {f1_score(results['true'], results['preds'], average='weighted')}')

cm = confusion_matrix(results['true'], results['preds'])
ConfusionMatrixDisplay(cm).plot()

# %% [markdown]
# # Crossvalidation

# %%
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# accuracies = []


# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# for train_index, test_index in skf.split(data, labels):
    
#     X_train_fold, X_test = data[train_index], data[test_index]
#     Y_train_fold, Y_test = labels[train_index], labels[test_index]

#     X_train_aug = []
#     Y_train_aug = []

#     for i, arr in enumerate(X_train_fold):
#         for _ in range(2):
#             X_train_aug.append(arr + np.random.normal(0, 0.001, size=arr.shape))
#             Y_train_aug.append(Y_train_fold[i])

#     X_train_aug = np.array(X_train_aug)
#     Y_train_aug = np.array(Y_train_aug)

#     X_train_fold = np.concatenate([X_train_fold, X_train_aug], axis=0)
#     Y_train_fold = np.concatenate([Y_train_fold, Y_train_aug], axis=0)

#     print(X_train_fold.shape)

#     model = make_model()
#     model.load_weights('started.weights.h5')

#     model.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Nadam(),
#                 metrics=['accuracy']
#     )

#     mc = tf.keras.callbacks.ModelCheckpoint('best_model_folds.keras', monitor='val_loss', mode='min', save_best_only=True)

#     preprocess.fit(X_train)
#     history = model.fit(
#         preprocess.transform(X_train_fold),
#         tf.keras.utils.to_categorical(Y_train_fold),
#         batch_size=8,
#         epochs=1,
#         verbose=0,
#         callbacks=[mc]
#     )

#     print(len(Y_test))

#     model = tf.keras.models.load_model('best_model_folds.keras')

#     preds = np.argmax(model.predict(preprocess.transform(X_test), verbose=0), axis=1)

#     acc = accuracy_score(Y_test, preds)
#     accuracies.append(acc)

#     print(f'Test accuracy: {acc}')
#     print(f'Test recall: {recall_score(Y_test, preds, average='weighted')}')
#     print(f'Test precision: {precision_score(Y_test, preds, average='weighted')}')
#     print(f'Test f1: {f1_score(Y_test, preds, average='weighted')}')

#     cm = confusion_matrix(Y_test, preds)
#     print(cm)
#     print()
    

# print(f'Mean test accuracy: {np.mean(accuracies)}')


