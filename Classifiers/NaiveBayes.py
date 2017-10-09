from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

class NaiveBayes:
    # baeta við kfold i parameters?
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.kValues = list(range(1, len(self.X_train[0]) + 1)) # telja columns i x_train
        self.kFold = KFold(n_splits=2)

        self.grid_scores = ""
        self.best_score = ""
        self.best_params = ""

        self.getOptimalKParameter()

    def getOptimalKParameter(self):
        top_feat = SelectKBest(chi2)

        pipe = Pipeline([('feat', top_feat),
                         ('clf', GaussianNB(priors=None))])

        param_grid = [{'feat__k': self.kValues}]
        scoring = 'accuracy'

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=2)
        gs.fit(self.X_train, self.y_train)

        self.grid_scores = gs.grid_scores_

        self.best_score = gs.best_score_
        self.best_params = gs.best_params_

    def printBestScoreAndParam(self):
        print("Naive Bayes\n------------\n" +
              "Best Parameter: " + str(self.best_params) + "\n"+
              "Best Score: " + str(self.best_score) + "\n")