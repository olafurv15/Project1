from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

class NeuralNetworks:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.bestActivation = 0
        self.bestLearningRate = 0
        self.score = 0

        self.activationFunctions = ['identity', 'logistic', 'tanh', 'relu']
        self.learningRates = np.arange(0.0001, 0.01, 0.0001)
        self.network = MLPClassifier()

        self.getOptimalParameters()
        self.computeScore()

    def getOptimalParameters(self):
        pipe = Pipeline([('clf', self.network)])

        param_grid = [{'clf__activation': self.activationFunctions,
                       'clf__learning_rate_init': self.learningRates}]

        scoring = 'accuracy'

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=2)
        gs.fit(self.X_train, self.y_train)

        self.bestActivation = gs.best_params_['clf__activation']
        self.bestLearningRate = gs.best_params_['clf__learning_rate_init']

    def computeScore(self):
        network = MLPClassifier(learning_rate_init=self.bestLearningRate, activation=self.bestActivation)
        network.fit(self.X_train, self.y_train)
        self.score = network.score(self.X_test, self.y_test)

    def getScore(self):
        return self.score

    def getClassifierName(self):
        return "Neural Networks"

    def printBestScoreAndParam(self):
        print("\nNeural Networks\n-------------------\n" +
              "Best Learning Rate: " + str(self.bestLearningRate) + "\n"+
              "Best Activation: " + str(self.bestActivation) + "\n" +
              "Score: " + str(self.score) + "\n")