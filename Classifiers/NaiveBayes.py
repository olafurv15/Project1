from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

class NaiveBayes:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.kValues = list(range(1, len(self.X_train[0]) + 1))

        self.score = 0
        self.best_k = 0

        self.getOptimalKParameter()
        self.computeScore()

    def getOptimalKParameter(self):
        top_feat = SelectKBest(chi2)

        pipe = Pipeline([('feat', top_feat),
                         ('clf', GaussianNB(priors=None))])

        param_grid = [{'feat__k': self.kValues}]
        scoring = 'accuracy'

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=2)
        gs.fit(self.X_train, self.y_train)

        self.best_k = gs.best_params_['feat__k']

        print(gs.grid_scores_)

    def computeScore(self):
        feature_selector = SelectKBest(chi2, k=self.best_k)
        X_train_trans = feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_trans = feature_selector.transform(self.X_test)

        clf = GaussianNB(priors=None)
        clf.fit(X_train_trans, self.y_train)
        self.score = clf.score(X_test_trans, self.y_test)

    def getScore(self):
        return self.score

    def getClassifierName(self):
        return "Naive Bayes"

    def printBestScoreAndParam(self):
        print("\nNaive Bayes\n-------------------\n" +
              "Best K Parameter: " + str(self.best_k) + "\n" +
              "Score: " + str(self.score) + "\n")