import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from scipy import stats

class NeuralNetworks:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.activationFunctions = ['identity', 'logistic', 'tanh', 'relu']
        self.learningRates = np.arange(0.0001, 0.01, 0.0001)
        self.kf = KFold(n_splits=2)

        self.resTrainAccuracy2 = []
        self.resValidAccuracy2 = []
        self.resTrainError2 = []
        self.resValidError2 = []

        self.resTrainAccuracy3 = []
        self.resValidAccuracy3 = []
        self.resTrainError3 = []
        self.resValidError3 = []


    def getActivationFunction(self):
        for function in self.activationFunctions:
            trainAccuracy = []
            validationAccuracy = []
            for train_index, test_index in self.kf.split(self.X_train):
                x_cv_train = [self.X_train[i] for i in train_index]
                y_cv_train = [self.y_train[i] for i in train_index]
                x_cv_test = [self.X_train[i] for i in test_index]
                y_cv_test = [self.y_train[i] for i in test_index]
                Network = MLPClassifier(activation=function)
                Network.fit(x_cv_train, y_cv_train)
                trainAccuracy.append(Network.score(x_cv_train, y_cv_train))
                validationAccuracy.append(Network.score(x_cv_test, y_cv_test))

            self.resTrainAccuracy2.append(np.mean(trainAccuracy) * 100)
            self.resValidAccuracy2.append(np.mean(validationAccuracy) * 100)
            self.resTrainError2.append(stats.sem(trainAccuracy) * 100)
            self.resValidError2.append(stats.sem(validationAccuracy) * 100)

    def getLearningRate(self):
        for rate in self.learningRates:
            trainAccuracy = []
            validationAccuracy = []
            for train_index, test_index in self.kf.split(self.X_train):
                x_cv_train = [self.X_train[i] for i in train_index]
                y_cv_train = [self.y_train[i] for i in train_index]
                x_cv_test = [self.X_train[i] for i in test_index]
                y_cv_test = [self.y_train[i] for i in test_index]
                Network = MLPClassifier(learning_rate_init=rate)
                Network.fit(x_cv_train, y_cv_train)
                trainAccuracy.append(Network.score(x_cv_train, y_cv_train))
                validationAccuracy.append(Network.score(x_cv_test, y_cv_test))

            self.resTrainAccuracy3.append(np.mean(trainAccuracy) * 100)
            self.resValidAccuracy3.append(np.mean(validationAccuracy) * 100)
            self.resTrainError3.append(stats.sem(trainAccuracy) * 100)
            self.resValidError3.append(stats.sem(validationAccuracy) * 100)

    def getScore(self):
        pass
        #Network = MLPClassifier(learning_rate_init=0.0001, activation='relu')
        #Network.fit(X_train, y_train)
        #Network.score(X_test, y_test)