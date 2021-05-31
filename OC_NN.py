from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
#load the training dataset
def load_data(file,activity):
    column_names = ['timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']

    df = pd.read_csv(file,
                     header=None,
                     names=column_names)

    if activity == 1:
      df['activity'] = 1
    elif activity == 0 :
      df['activity'] = 0
    
    df['x-axis'].replace(regex=True,
      inplace=True,
      to_replace=r' ',
      value=r'')

    df['y-axis'].replace(regex=True,
      inplace=True,
      to_replace=r' ',
      value=r'')
    
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r' ',
      value=r'')

    df['z-axis'] = df['z-axis'].apply(convert_to_float)

    df.dropna(axis=0, how='any', inplace=True)

    return df
#Convert a number to a float
def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
#Split x and y
def split_x_y(df, time_steps, step, label_name):

    N_FEATURES = 3

    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels
#Building confusion matrix
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#LABELS
LABELS = ['Anomaly','Activity']

#load dataset train and test
data_train = load_data('activity_std.txt',1)
data_activity = load_data('data_test.txt',1)
data_fall = load_data('data_test_fall.txt',0)

frames = [data_activity, data_fall]
data_test = pd.concat(frames)

#Normalise data
pd.options.mode.chained_assignment = None  # default='warn'
data_train['x-axis'] = data_train['x-axis'] / data_train['x-axis'].max()
data_test['x-axis'] = data_test['x-axis'] / data_test['x-axis'].max()
data_train['y-axis'] = data_train['y-axis'] / data_train['y-axis'].max()
data_test['y-axis'] = data_test['y-axis'] / data_test['y-axis'].max()
data_train['z-axis'] = data_train['z-axis'] / data_train['z-axis'].max()
data_test['z-axis'] = data_test['z-axis'] / data_test['z-axis'].max()

data_train = data_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
data_test = data_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

#split the data
TIME_PERIODS = 5
STEP_DISTANCE = 2
LABEL = 'activity'

x_train, y_train = data_train.iloc[:, 0:3] , data_train.iloc[:,4]
x_test, y_test = data_test.iloc[:, 0:3] , data_test.iloc[:,4]

#x_train, y_train = split_x_y(data_train,TIME_PERIODS,STEP_DISTANCE,LABEL)
#x_test, y_test = split_x_y(data_test,TIME_PERIODS,STEP_DISTANCE,LABEL)

# Set input & output dimensions
num_sensors = x_train.shape[1]
num_classes = 2

input_shape = (num_sensors)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#Architecture of the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_shape,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.55),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

#Loss function
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#Parametres
EPOCHS = 20
BATCH_SIZE = 400

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1)
]
#training the model
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

print("Test : ")
#evaluate the model
model.evaluate(x_test,y_test)

y_pred_test = model.predict(x_test)
y_pred_test = np.argmax(y_pred_test, axis=1)
y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(y_test, y_pred_test)

print(classification_report(y_test, y_pred_test))