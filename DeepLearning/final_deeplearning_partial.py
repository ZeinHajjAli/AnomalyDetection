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

dataset_train = pd.read_csv('data/UNSW_NB15_training-set.csv')
dataset_train.head()
dataset_test = pd.read_csv('data/UNSW_NB15_testing-set.csv')
dataset_test.head()

dataset_train['attack_cat'].unique()

dataset_train['label'].unique()

del dataset_train['id']
del dataset_test['id']
del dataset_train['attack_cat']
del dataset_test['attack_cat']

dataset = pd.concat([dataset_train, dataset_test])
print(dataset.head())
print(dataset.shape)

cat_features = dataset.select_dtypes(include=['category', object]).columns
print(cat_features)

# Convert category types to numeric
dataset[cat_features] = dataset[cat_features].apply(LabelEncoder().fit_transform)
dataset.head()

# dataset_train, dataset_test = train_test_split(dataset, test_size=0.3)
# X is input features, Y is target features

dataset_train, dataset_test = train_test_split(dataset, test_size=0.4, stratify=dataset['label'])
dataset_test, dataset_val = train_test_split(dataset_test, test_size=0.25, stratify=dataset_test['label'])

# ensure all data are floating point values
# X = X.astype('float32')

# Normalize/Standardize data

X_train = dataset_train.loc[:, dataset.columns != 'label']
Y_train = dataset_train['label']

X_test = dataset_test.loc[:, dataset.columns != 'label']
Y_test = dataset_test['label']

X_val = dataset_val.loc[:, dataset.columns != 'label']
Y_val = dataset_val['label']

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_val_scaled = scaler.fit_transform(X_val)


VAL_ACC = []
VAL_LOSS = []
model_filename = 'sequential.model'
save_dir = 'model/'

# Construct the NN
model = Sequential()
model.add(Dense(10, input_dim = 42, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile
model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(save_dir+'deepLearning_partial_checkpoint.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

start_train = time.perf_counter()
history = model.fit(X_train_scaled, Y_train, epochs = 10, callbacks = callbacks_list, validation_data= (X_val_scaled, Y_val))
end_train = time.perf_counter()
training_time = end_train - start_train

model.load_weights(save_dir+"deepLearning_partial_checkpoint.h5")
model.save(save_dir + 'deepLearning_partial.h5')

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
    fig.savefig('figures/DL_partial_acc.png')

    fig = plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    fig.savefig('figures/DL_partial_loss.png')


plot_training(history)
