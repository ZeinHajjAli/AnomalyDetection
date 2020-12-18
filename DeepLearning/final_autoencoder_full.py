import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from sklearn.metrics import classification_report
from sklearn import metrics
import time

dataset = pd.read_csv('data/combined.csv')
dataset.head()

dataset.columns = dataset.columns.str.replace(' ', '')
dataset['srcip'] = dataset['srcip'].str.strip()
dataset['sport'] = dataset['sport'].str.strip()
dataset['dstip'] = dataset['dstip'].str.strip()
dataset['dsport'] = dataset['dsport'].str.strip()
dataset['proto'] = dataset['proto'].str.strip()
dataset['state'] = dataset['state'].str.strip()
dataset['service'] = dataset['service'].str.strip()
dataset['ct_ftp_cmd'] = dataset['ct_ftp_cmd'].str.strip()
dataset.loc[(dataset.ct_ftp_cmd.isnull()), 'ct_ftp_cmd'] = 0
dataset.loc[(dataset.ct_ftp_cmd == ''), 'ct_ftp_cmd'] = 0
dataset['ct_ftp_cmd'] = dataset['ct_ftp_cmd'].astype(int)
dataset['attack_cat'] = dataset['attack_cat'].str.strip()
dataset.loc[(dataset.Label == 0), 'attack_cat'] = 'Normal'

dataset.drop_duplicates()

indexNames = dataset[dataset['srcip'] == '127.0.0.1'].index
dataset.drop(indexNames, inplace=True)
dataset['srcip'].unique()

dataset.loc[(dataset['service'] == '-'), 'service'] = ''
dataset.loc[(dataset['sport'].isnull()), 'sport'] = ''
dataset.loc[(dataset['dsport'].isnull()), 'dsport'] = ''
print(dataset['srcip'].unique())

print('anom: ', dataset.loc[(dataset['Label'] == 1)].shape[0])
print('norm: ', dataset.loc[(dataset['Label'] == 0)].shape[0])

"""# **Remove 6 features**"""

# iterating the columns 
for col in dataset.columns: 
    print('Name:', col, ' Types:', dataset[col].dtypes, 'Null value:', dataset[col].isnull().sum())

dataset['service'].unique()

del dataset['srcip']
del dataset['dstip']
del dataset['sport']
del dataset['dsport']
del dataset['Stime']
del dataset['Ltime']

dataset['ct_flw_http_mthd'] = dataset['ct_flw_http_mthd'].fillna(0)
dataset['is_ftp_login'] = dataset['is_ftp_login'].fillna(0)

dataset['ct_flw_http_mthd'].unique()

dataset['ct_flw_http_mthd'].unique()

for col in dataset.columns: 
    print('Name:', col, ' Types:', dataset[col].dtypes, 'Null value:', dataset[col].isnull().sum())

"""**Remove Unused features**"""
dataset_normal = dataset.loc[(dataset['Label'] == 0)]
dataset_attack = dataset.loc[(dataset['Label'] == 1)]
_, dataset_normal = train_test_split(dataset_normal, test_size=0.1463767002, stratify=dataset_normal['attack_cat'])
print('norm: ', dataset_normal.shape[0])
print('anom: ', dataset_attack.shape[0])

dataset_train_unsupervised, dataset_normal_remaining = train_test_split(dataset_normal, test_size=0.6)
dataset_remaining = pd.concat([dataset_normal_remaining, dataset_attack])
dataset_train_supervised, dataset_test = train_test_split(dataset_remaining, test_size=0.25)

del dataset_train_unsupervised['attack_cat']
del dataset_train_supervised['attack_cat']
del dataset_test['attack_cat']

"""**AUTO ENCODER PART**"""

cat_features = dataset_train_unsupervised.select_dtypes(include=['category', object]).columns
print(cat_features)

# Convert category types to numeric
dataset_train_unsupervised[cat_features] = dataset_train_unsupervised[cat_features].apply(LabelEncoder().fit_transform)
Y_AE_train = dataset_train_unsupervised['Label']
X_AE_train = dataset_train_unsupervised.loc[:, dataset_train_unsupervised.columns != 'Label']

dataset_train_unsupervised.head()

dataset_train_unsupervised['service'].unique()

# ensure all data are floating point values
X_AE_train = X_AE_train.astype('float32')

# Normalize/Standardize data

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
autoencoder.fit(X_AE_train_scaled[np.where(Y_AE_train == 0)], X_AE_train_scaled[np.where(Y_AE_train == 0)],
                epochs=100, validation_split=0)
end_train_AE = time.perf_counter()
training_time_AE = end_train_AE - start_train_AE

"""**Deep learning Feed-Forward Neuron Network**"""

cat_features = dataset_train_supervised.select_dtypes(include=['category', object]).columns
print(cat_features)

"""**Convert to numeric Trainning Set**"""

# Convert category types to numeric
dataset_train_supervised[cat_features] = dataset_train_supervised[cat_features].apply(LabelEncoder().fit_transform)
Y_FF_train = dataset_train_supervised['Label']
X_FF_train = dataset_train_supervised.loc[:, dataset_train_supervised.columns != 'Label']

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
Y_FF_test = dataset_test['Label']
X_FF_test = dataset_test.loc[:, dataset_test.columns != 'Label']

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

for i in range(len(temp_weights)-1):
    classifier_NN.layers[i].set_weights(temp_weights[i])


classifier_NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train_FF = time.perf_counter()
history = classifier_NN.fit(X_FF_train_scaled, Y_FF_train, epochs=100, validation_split=0)
end_train_FF = time.perf_counter()
training_time_FF = end_train_FF - start_train_FF

save_dir = 'model/'
classifier_NN.save(save_dir+'autoencoder_full.h5')

# classifier_NN.load_weights(save_dir+"final_model.h5")

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
      # val_acc = history.history['val_accuracy']
      loss = model_history.history['loss']
      # val_loss = history.history['val_loss']
      epochs = range(len(acc))

      plt.plot(epochs, acc, 'r.')
      # plt.plot(epochs, val_acc, 'r')
      plt.title('Training accuracy')

      plt.figure()
      plt.plot(epochs, loss, 'r.')
      # plt.plot(epochs, val_loss, 'r-')
      plt.title('Training loss')
      plt.show()
  
plot_training(history)