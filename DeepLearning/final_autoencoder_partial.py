import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import time

dataset_train = pd.read_csv('data/UNSW_NB15_training-set.csv')
dataset_train.head()
dataset_test = pd.read_csv('data/UNSW_NB15_testing-set.csv')
dataset_test.head()

"""
Remove Unused features

Full Dataset
"""

# Full dataset
dataset = pd.concat([dataset_train, dataset_test])
print(dataset.head())
print(dataset.shape)
print(dataset.loc[(dataset['label'] == 1)].shape)
print(dataset.loc[(dataset['label'] == 0)].shape)

dataset_normal = dataset.loc[(dataset['label'] == 0)]
dataset = dataset.loc[(dataset['label'] == 1)]
dataset_attack, _ = train_test_split(dataset, test_size=0.441317034, stratify=dataset['attack_cat'])

dataset_train_unsupervised, dataset_normal_remaining = train_test_split(dataset_normal, test_size=0.6)
dataset_remaining = pd.concat([dataset_normal_remaining, dataset_attack])
dataset_train_supervised, dataset_test = train_test_split(dataset_remaining, test_size=0.25)

del dataset_train_unsupervised['id']
del dataset_train_supervised['id']
del dataset_test['id']
del dataset_train_unsupervised['attack_cat']
del dataset_train_supervised['attack_cat']
del dataset_test['attack_cat']

"""AUTO ENCODER PART"""

cat_features = dataset_train_unsupervised.select_dtypes(include=['category', object]).columns
print(cat_features)

# Convert category types to numeric
dataset_train_unsupervised[cat_features] = dataset_train_unsupervised[cat_features].apply(LabelEncoder().fit_transform)
Y_AE_train = dataset_train_unsupervised['label']
X_AE_train = dataset_train_unsupervised.loc[:, dataset_train_unsupervised.columns != 'label']

# ensure all data are floating point values
X_AE_train = X_AE_train.astype('float32')

# Standardize data
scaler = StandardScaler()
X_AE_train_scaled = scaler.fit_transform(X_AE_train)

input_data = Input(shape=(X_AE_train_scaled.shape[1],))
encoded = Dense(units=10, activation='tanh')(input_data)
encoded = Dense(units=3, activation='tanh')(encoded)
decoded = Dense(units=10, activation='tanh')(encoded)
decoded = Dense(units=X_AE_train_scaled.shape[1], activation='tanh')(decoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

start_train_AE = time.perf_counter()
autoencoder.fit(X_AE_train_scaled[np.where(Y_AE_train == 0)], X_AE_train_scaled[np.where(Y_AE_train == 0)], epochs=100, validation_split=0)
end_train_AE = time.perf_counter()
training_time_AE = end_train_AE - start_train_AE

"""**Deep learning Feed-Forward Neuron Network**"""

cat_features = dataset_train_supervised.select_dtypes(include=['category', object]).columns
print(cat_features)

"""**Convert to numeric Training Set**"""

# Convert category types to numeric
dataset_train_supervised[cat_features] = dataset_train_supervised[cat_features].apply(LabelEncoder().fit_transform)
Y_FF_train = dataset_train_supervised['label']
X_FF_train = dataset_train_supervised.loc[:, dataset_train_supervised.columns != 'label']

print(Y_FF_train.shape)
print(X_FF_train.shape)

# ensure all data are floating point values
X_FF_train = X_FF_train.astype('float32')

# Standardize data
scaler = StandardScaler()
X_FF_train_scaled = scaler.fit_transform(X_FF_train)
print(X_FF_train_scaled.shape)

"""**Convert to numeric Testing Set**"""

# Convert category types to numeric
dataset_test[cat_features] = dataset_test[cat_features].apply(LabelEncoder().fit_transform)
Y_FF_test = dataset_test['label']
X_FF_test = dataset_test.loc[:, dataset_test.columns != 'label']

print(Y_FF_test.shape)
print(X_FF_test.shape)

# ensure all data are floating point values
X_FF_test = X_FF_test.astype('float32')

# Standardize data
scaler = StandardScaler()
X_FF_test_scaled = scaler.fit_transform(X_FF_test)
print(X_FF_test_scaled.shape)

"""**Using AUTOENCODER to PREDICT THE X_FF_TRAIN**"""

X_FF_train_predict = autoencoder.predict(X_FF_train_scaled)
X_FF_test_predict = autoencoder.predict(X_FF_test_scaled)

"""**CONSTRUCT THE NETWORK**"""

print(X_FF_test_predict.shape)
print(Y_FF_test.shape)
print(X_FF_train_predict.shape)
print(Y_FF_train.shape)

# Construct the NN
temp_weights = [layer.get_weights() for layer in autoencoder.layers]

input_layer = Input(shape=(X_FF_train_scaled.shape[1],))
encoded_layer = Dense(units=10, activation='tanh')(input_layer)
encoded_layer = Dense(units=3, activation='tanh')(encoded_layer)
decoded_layer = Dense(units=10, activation='tanh')(encoded_layer)
decoded_layer = Dense(units=1, activation='sigmoid')(decoded_layer)

classifier_NN = Model(input_layer, decoded_layer)

for i in range(len(temp_weights) - 1):
    classifier_NN.layers[i].set_weights(temp_weights[i])

classifier_NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train_FF = time.perf_counter()
history = classifier_NN.fit(X_FF_train_scaled, Y_FF_train, epochs=100, validation_split=0)
end_train_FF = time.perf_counter()
training_time_FF = end_train_FF - start_train_FF

save_dir = 'model/'
classifier_NN.save(save_dir + 'autoencoder_partial.h5')

start_eval = time.perf_counter()
score = classifier_NN.evaluate(X_FF_test_scaled, Y_FF_test)
end_eval = time.perf_counter()
eval_time = end_eval - start_eval

print(score)
y_pred = classifier_NN.predict(X_FF_test_scaled)
y_pred = (y_pred > 0.5)
print(classification_report(Y_FF_test, y_pred))
conf_matrix = metrics.confusion_matrix(Y_FF_test, y_pred)
print(conf_matrix)

start_pred = time.perf_counter()
_ = classifier_NN.predict(X_FF_test_scaled)
end_pred = time.perf_counter()
pred_time = end_pred - start_pred

print('AE Training time: ', training_time_AE)
print('FF Training time: ', training_time_FF)
training_time_combined = training_time_AE + training_time_FF
print('Combined Training time: ', training_time_combined)
print('Evaluation time: ', eval_time)
print('Prediction time: ', pred_time)
print('Test set shape: ', X_FF_test_scaled.shape)


def plot_training(model_history):
    acc = model_history.history['accuracy']
    loss = model_history.history['loss']
    epochs = range(len(acc))

    fig = plt.figure()
    plt.plot(epochs, acc, 'r.')
    plt.title('Training accuracy')
    fig.savefig('figures/AE_partial_acc.png')

    fig = plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.title('Training loss')
    fig.savefig('figures/AE_partial_loss.png')


plot_training(history)
