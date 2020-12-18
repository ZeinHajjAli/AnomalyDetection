import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
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

print(dataset['srcip'].unique())
print(dataset['sport'].unique())
print(dataset['dstip'].unique())
print(dataset['dsport'].unique())
print(dataset['proto'].unique())
print(dataset['state'].unique())
print(dataset['service'].unique())
print(dataset['attack_cat'].unique())

indexNames = dataset[dataset['srcip'] == '127.0.0.1'].index
dataset.drop(indexNames, inplace=True)
dataset['srcip'].unique()

dataset.loc[(dataset['service'] == '-'), 'service'] = ''
dataset.loc[(dataset['sport'].isnull()), 'sport'] = ''
dataset.loc[(dataset['dsport'].isnull()), 'dsport'] = ''

dataset.head()

del dataset['srcip']
del dataset['dstip']
del dataset['attack_cat']
del dataset['dbytes']
del dataset['ct_dst_ltm']
del dataset['sbytes']
del dataset['Sintpkt']
del dataset['ct_flw_http_mthd']
del dataset['res_bdy_len']
del dataset['is_sm_ips_ports']
del dataset['Dintpkt']
del dataset['ct_ftp_cmd']
del dataset['is_ftp_login']
del dataset['sport']
del dataset['ct_src_dport_ltm']

cat_features = dataset.select_dtypes(include=['category', object]).columns
print(cat_features)

# Convert category types to numeric
dataset[cat_features] = dataset[cat_features].apply(LabelEncoder().fit_transform)
dataset.head()

dataset_train, dataset_test = train_test_split(dataset, test_size=0.4, stratify=dataset['Label'])
dataset_test, dataset_val = train_test_split(dataset_test, test_size=0.25, stratify=dataset_test['Label'])

# ensure all data are floating point values
# X = X.astype('float32')

# Normalize/Standardize data

X_train = dataset_train.loc[:, dataset.columns != 'Label']
Y_train = dataset_train['Label']

X_test = dataset_test.loc[:, dataset.columns != 'Label']
Y_test = dataset_test['Label']

X_val = dataset_val.loc[:, dataset.columns != 'Label']
Y_val = dataset_val['Label']

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_val_scaled = scaler.fit_transform(X_val)

VAL_ACC = []
VAL_LOSS = []
save_dir = 'model/'

# Construct the NN
model = Sequential()
model.add(Dense(10, input_dim = 33, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile
model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(save_dir+'deepLearning_full_checkpoint.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

start_train = time.perf_counter()
history = model.fit(X_train_scaled, Y_train, epochs = 10, callbacks = callbacks_list, validation_data= (X_val_scaled, Y_val))
end_train = time.perf_counter()
training_time = end_train - start_train

model.load_weights(save_dir+"deepLearning_full_checkpoint.h5")
model.save(save_dir + 'deepLearning_full.h5')

start_eval = time.perf_counter()
score = model.evaluate(X_test_scaled, Y_test)
end_eval = time.perf_counter()
eval_time = end_eval - start_eval

print(score)
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5)
print(classification_report(Y_test, y_pred))
conf_matrix = metrics.confusion_matrix(Y_test, y_pred)
print(conf_matrix)

start_pred = time.perf_counter()
_ = model.predict(X_test_scaled)
end_pred = time.perf_counter()
pred_time = end_pred - start_pred

print('Training time: ', training_time)
print('Evaluation time: ', eval_time)
print('Prediction time: ', pred_time)
print('Test set shape: ', X_test_scaled.shape)


def plot_training(model_history):
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(acc))

    fig = plt.figure()
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    fig.savefig('figures/DL_full_acc.png')

    fig = plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    fig.savefig('figures/DL_full_acc.png')


plot_training(history)
