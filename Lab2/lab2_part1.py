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


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


M = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(M)
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
#First, we just use a single neuron. 


#####TO COMPLETE

def accuracy(labels, predictions):
    return np.sum(labels == predictions) / labels.shape[1]

def fScore(labels, predictions):
    cmat = confusion_matrix(labels[0],predictions[0])    

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

class Neuron():
    def __init__(self, input_dim, activation = 'sigmoid'):
        self.w = np.random.randn(input_dim, 1) * 0.01
        self.b = 0
        self.activation = activation

    def activationFn(self):
        if self.activation == 'sigmoid':
            return lambda x : 1 / (1 + np.exp(-x))
        return None

    def forward(self, x):
        z = np.dot(self.w.T, x) + self.b
        a = self.activationFn()(z)
        return a  

    def loss(self, y, a):
        m = y.shape[1]
        cost = -1/m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))  
        return cost
    
    def loss_derivative(self, x, y, a):
        m = x.shape[1]
        dw = np.dot(x, (a - y).T) / m
        db = np.sum(a - y) / m
        return dw, db

    def backprop(self, x, y, a):
        m = x.shape[1]
        dw, db = self.loss_derivative(x, y, a)        

        grads = {"dw": dw,
                 "db": db}

        return grads

    def train(self, x_train, y_train, x_valid, y_valid, metrics, num_iterations, learning_rate, print_cost = True):
        history = dict()
        history['loss_train'] = []
        history['loss_valid'] = []
        for k in range(len(metrics)):
            history["metric" + str(k) + "_train"] = []
            history["metric" + str(k) + "_valid"] = []

        for i in range(num_iterations):
            # Cost and gradient calculation
            # training
            a_train = self.forward(x_train)
            cost_train = self.loss(y_train, a_train)
            grads = self.backprop(x_train, y_train, a_train)
            y_prediction = (a_train > 0.5) * 1

            history['loss_train'].append(cost_train)
            for k in range(len(metrics)):
                history["metric" + str(k) + "_train"].append(metrics[k](y_train, y_prediction))

            # validation
            a_valid = self.forward(x_valid)
            cost_valid = self.loss(y_valid, a_valid)
            y_prediction = (a_valid > 0.5) * 1

            history['loss_valid'].append(cost_valid)
            for k in range(len(metrics)):
                history["metric" + str(k) + "_valid"].append(metrics[k](y_valid, y_prediction))
            
            # update rule
            self.w = self.w - learning_rate * grads["dw"]
            self.b = self.b - learning_rate * grads["db"]            
            
            # Print the cost every 100 training iterations
            if print_cost and i % 10 == 0:
                print ("Cost after iteration " + str(i) + ": loss_train: " + str(history['loss_train'][i]) + ": loss_valid: " + str(history['loss_valid'][i]))

        return history

    def predict(self, x):
        m = x.shape[1]
        y_prediction = np.zeros((1,m))
        
        a = self.forward(x)        
        y_prediction = (a > 0.5) * 1
        
        assert(y_prediction.shape == (1, m))
        
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
        np.savez(file, w=self.w, b=self.b)

    def loadWeights(self, file):
        data = np.load(file)
        self.w = data['w']
        self.b = data['b']


neuron = Neuron(x_train.shape[0])
pretrained = False

if pretrained:
    neuron.loadWeights('weigths.npz')
else:
    history = neuron.train(x_train, y_train, x_valid, y_valid, [accuracy, fScore, bacc], 10, 1)
    # neuron.saveWeights('weigths.npz')

    fig = plt.figure()
    plt.plot(history['loss_train'], label="loss_train")
    plt.plot(history['loss_valid'], label="loss_valid")
    plt.xlabel('iteration')
    plt.ylabel('Losses')
    plt.legend()
    plt.title('loss_train vs loss_valid')
    # fig.savefig('loss_single.png', dpi=fig.dpi)
    plt.show()

    fig = plt.figure()
    plt.plot(history['metric0_train'], label="accuracy_train")
    plt.plot(history['metric1_train'], label="f_score_train")
    plt.plot(history['metric2_train'], label="bacc_train")
    plt.plot(history['metric0_valid'], label="accuracy_valid")
    plt.plot(history['metric1_valid'], label="f_score_valid")
    plt.plot(history['metric2_valid'], label="bacc_valid")
    plt.xlabel('iteration')
    plt.ylabel('metric value')
    plt.legend()
    plt.title('Metrics')
    # fig.savefig('metric_single.png', dpi=fig.dpi)
    plt.show()

test_scores = neuron.evaluate(x_test, y_test, [accuracy, fScore, bacc])
print(test_scores["metric0_test"], test_scores["metric1_test"], test_scores["metric2_test"])

# printing the wrongly classified instances

# predictions = neuron.predict(x_test)
# wrong = np.argwhere(predictions != y_test)
# fig = plt.figure(figsize = (12,12))
# count = 1
# for index in wrong:
#    plt.subplot(5,5,count) 
#    plt.gca().set_title(int(predictions[:,index[1]]), color = 'red')
#    plt.imshow(X_test[:,index[1]].reshape(28,28), cmap = plt.cm.binary)
#    plt.axis("off")
#    if count == 25:
#      break
#    count +=1

