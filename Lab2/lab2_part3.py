from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#In this first part, we just prepare our data (mnist) 
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
x_test  = X_test / 255



# one-hot encode labels
digits = 10

def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new

y_train=one_hot_encode(y_train, digits)
y_test=one_hot_encode(y_test, digits)

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(X_train.shape[1])
x_shuffled, y_shuffled = X_train[:,shuffle_index], y_train[:,shuffle_index]
x_train, y_train = x_shuffled[:,:50000], y_shuffled[:,:50000]
x_valid, y_valid = x_shuffled[:,50000:], y_shuffled[:,50000:]

# #Display one image and corresponding label 
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network

#####TO COMPLETE

def accuracy(labels, predictions):
    return np.sum(np.argmax(labels, axis=0) == predictions) / labels.shape[1]

def fScore(labels, predictions):
    cmat = confusion_matrix(np.argmax(labels, axis=0)[0],predictions[0])    

    TP = cmat[0][0]
    FP = cmat[0][1]
    FN = cmat[1][0]
    TN = cmat[1][1]
    
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP/ (TP + FN)
    return (2*precision * recall)/(precision + recall)

def bacc(labels, predictions):
    cmat = confusion_matrix(labels[0],predictions[0])    

    TP = cmat[0][0]
    FP = cmat[0][1]
    FN = cmat[1][0]
    TN = cmat[1][1]
    
    return  ((TP/(TP+FP)) + (TN/(TN+FN)))/2

# simple neural network with one hidden layer
class NeuralNetworkMultiClass():
    def __init__(self, input_shape, hidden_layer_size, output_shape):
        n_x = input_shape[0]
        n_h = hidden_layer_size
        n_y = output_shape[0]

        self.W1 = np.random.randn(n_h, n_x) * 0.01
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(n_y, n_h) * 0.01
        self.b2 = np.zeros((n_y, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, x, activation):
        Z1 = np.dot(self.W1, x) + self.b1
        # tanh activation for hidden layer
        A1 = np.tanh(Z1)
        # sigmoid activation for hidden layer
        # A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = activation(Z2)

        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}

        return A2, cache

    def loss(self, y, a):
        m = y.shape[1]
        cost = -1/m * np.sum(y * np.log(a))
        cost = float(np.squeeze(cost))
        return cost
    
    def loss_derivative(self, x, y, a):
        m = x.shape[1]
        dw = np.dot(x, (a - y).T) / m
        return dw

    def backprop(self, x, y, a, cache):
        m = x.shape[1]
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        Z2 = cache['Z2']

        # output
        dZ2 = A2 - y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # hidden layer
        # backprop if using tanh activation for hidden layer
        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, x.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims = True) / m

        # backprop if using sigmoid activation for hidden layer
        # dZ1 = np.dot(self.W2.T, dZ2) * self.sigmoid_derivative(self.sigmoid(Z1))
        # dW1 = np.dot(dZ1, x.T) / m
        # db1 = np.sum(dZ1, axis=1, keepdims = True) / m

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}

        return grads

    def train(self, x_train, y_train, x_valid, y_valid, metrics, num_iterations, learning_rate, print_cost = True):
        history = dict()
        history['loss_train'] = []
        history['loss_valid'] = []
        for k in range(len(metrics)):
            history["metric" + str(k) + "_train"] = []
            history["metric" + str(k) + "_valid"] = []
        
        for i in range(num_iterations):
            # Cost and gradient calculation (≈ 1-4 lines of code)
            # training
            a_train, cache = self.forward(x_train, self.softmax)
            cost_train = self.loss(y_train, a_train)
            grads = self.backprop(x_train, y_train, a_train, cache)
            y_prediction = np.argmax(a_train, axis=0)

            history['loss_train'].append(cost_train)
            for k in range(len(metrics)):
                history["metric" + str(k) + "_train"].append(metrics[k](y_train, y_prediction))

            # validation
            a_valid, _ = self.forward(x_valid, self.softmax)
            cost_valid = self.loss(y_valid, a_valid)
            y_prediction = np.argmax(a_valid, axis=0)

            history['loss_valid'].append(cost_valid)
            for k in range(len(metrics)):
                history["metric" + str(k) + "_valid"].append(metrics[k](y_valid, y_prediction))
            
            # learning_rate = learning_rate * (1 / (1 + 0.0001 * i))

            # update rule
            self.W1 = self.W1 - learning_rate * grads['dW1']
            self.b1 = self.b1 - learning_rate * grads['db1']
            self.W2 = self.W2 - learning_rate * grads['dW2']
            self.b2 = self.b2 - learning_rate * grads['db2']
            
            # Print the cost every 10 training iterations
            if print_cost and i % 10 == 0:
                print ("Cost after iteration " + str(i) + ": loss_train: " + str(history['loss_train'][i]) + ": loss_valid: " + str(history['loss_valid'][i]))
        
        return history

    def predict(self, x):
        m = x.shape[1]
        y_prediction = np.zeros((1,m))        
        a, cache = self.forward(x, self.softmax)
        
        y_prediction = np.argmax(a, axis=0)        
        return y_prediction

    def evaluate(self, x_test, y_test, metrics):
        predictions = self.predict(x_test)
        evaluation = dict()
        for k in range(len(metrics)):
            evaluation["metric" + str(k) + "_test"] = []
        for k in range(len(metrics)):
            evaluation["metric" + str(k) + "_test"].append(metrics[k](y_test, predictions))
        return evaluation

    def saveWeights(self, file):
        np.savez(file, w1=self.W1, w2=self.W2, b1=self.b1, b2=self.b2)

    def loadWeights(self, file):
        data = np.load(file)
        self.W1 = data['w1']
        self.b1 = data['b1']
        self.W2 = data['w2']
        self.b2 = data['b2']


network_multi = NeuralNetworkMultiClass(X_train.shape, 64, y_train.shape)
pretrained = False

if pretrained:
    network_multi.loadWeights('SingleLayerMultiClassWeights.npz')
else:
    history = network_multi.train(x_train, y_train, x_valid, y_valid, [accuracy], 10, 1)
    # network_multi.saveWeights('SingleLayerMultiClassWeights.npz')

    fig = plt.figure()
    plt.plot(history['loss_train'], label="loss_train")
    plt.plot(history['loss_valid'], label="loss_valid")
    plt.xlabel('iteration')
    plt.ylabel('Losses')
    plt.legend()
    plt.title('loss_train vs loss_valid')
    # fig.savefig('loss_hidden_multiclass.png', dpi=fig.dpi)
    plt.show()

    fig = plt.figure()
    plt.plot(history['metric0_train'], label="accuracy_train")
    plt.plot(history['metric0_valid'], label="accuracy_valid")
    plt.xlabel('iteration')
    plt.ylabel('metric value')
    plt.legend()
    plt.title('Metrics')
    # fig.savefig('metric_hidden_multiclass.png', dpi=fig.dpi)
    plt.show()

test_scores = network_multi.evaluate(x_test, y_test, [accuracy])
print(test_scores["metric0_test"])

# printing the wrongly classified instances

# predictions = network_multi.predict(x_test)
# wrong = np.argwhere(predictions != np.argmax(y_test, axis=0))
# fig = plt.figure(figsize = (12,12))
# count = 1
# for index in wrong:
#     plt.subplot(5,5,count) 
#     plt.gca().set_title(int(predictions[index]), color = 'red')
#     plt.imshow(X_test[:,index].reshape(28,28), cmap = plt.cm.binary)
#     plt.axis("off")
#     if count == 25:
#       break
#     count +=1
# plt.show()

predictions = network_multi.predict(x_test)
cmat = confusion_matrix(np.argmax(y_test, axis=0), predictions, labels=[0, 1,2,3,4,5,6,7,8,9])
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import seaborn as sn
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    # print(cm)
    plt.figure(figsize=(14,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.png')
    plot_confusion_matrix(cmat, classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                          title='Confusion matrix')
