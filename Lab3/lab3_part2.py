import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


#load (first download if necessary) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0
y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


num_classes = 1


# custo accuracy functions
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


#Model architecture
network_binary = Sequential()
# parameters for the number of units: 16, 64, 1024
# possible activation functions: tanh, sigmoid, relu
network_binary.add(Dense(1024, input_shape=(x_train.shape[1],), activation='tanh'))
network_binary.add(Dense(num_classes, activation='sigmoid'))

#Compile and fit model
learning_rate = 0.01
network_binary.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy', f1])
history = network_binary.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2)

#Plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model metrics')
plt.ylabel('metric')
plt.xlabel('epoch')
plt.legend(['acc_train', 'acc_val', 'f1_train', 'f1_val'], loc='upper left')
plt.show()

#Test
network_binary.evaluate(x_test, y_test, batch_size=64, return_dict = True)