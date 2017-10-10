import numpy as np

from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.bestK = 0
        self.bestMinSplitParam = 0
        self.score = 0

        self.top_feat = SelectKBest(chi2)
        self.clfDTree = tree.DecisionTreeClassifier()

        self.kValues = np.arange(1, 10, 1)
        self.min_splitValues = np.arange(2, 1000, 10)

        self.getOptimalParameters()
        self.computeScore()

    def getOptimalParameters(self):
        pipe = Pipeline([('feat', self.top_feat),
                         ('clf', self.clfDTree)])

        param_grid = [{'feat__k': self.kValues,
                       'clf__min_samples_split': self.min_splitValues}]

        scoring = 'accuracy'

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=20)
        gs.fit(self.X_train, self.y_train)

        self.bestK = gs.best_params_['feat__k']
        self.bestMinSplitParam = gs.best_params_['clf__min_samples_split']


    def computeScore(self):
        # select features
        feature_selector = SelectKBest(chi2, k=self.bestK)
        x_cv_train = feature_selector.fit_transform(self.X_train, self.y_train)
        x_cv_test = feature_selector.transform(self.X_test)

        # train decision tree
        dt = tree.DecisionTreeClassifier(min_samples_split=self.bestMinSplitParam)
        dt.fit(x_cv_train, self.y_train)
        self.score = dt.score(x_cv_test, self.y_test)

    def getScore(self):
        return self.score

    def getClassifierName(self):
        return "Decision Tree"


    def printBestScoreAndParam(self):
        print("\nDecision Tree\n-------------------\n" +
              "Best K Parameter: " + str(self.bestK) + "\n"+
              "Best min samples split parameter: " + str(self.bestMinSplitParam) + "\n" +
              "Score: " + str(self.score) + "\n")
