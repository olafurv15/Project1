import numpy as np

from sklearn.model_selection import KFold
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.kValues = list(range(1, len(self.X_train[0]) + 1)) # telja columns i x_train
        self.kf = KFold(n_splits=2)

        self.resTrainAccuracy = []
        self.resValidAccuracy = []
        self.resTrainError = []
        self.resValidError = []

    def getKValues(self):
        # telja columns i x_train
        pass

    def getOptimalKParameter(self):
        for kParameter in self.kValues:
            trainAccuracy = []
            validationAccuracy = []
            for train_index, test_index in self.kf.split(self.X_train):
                x_cv_train = [self.X_train[i] for i in train_index]
                y_cv_train = [self.y_train[i] for i in train_index]
                x_cv_test = [self.X_train[i] for i in test_index]
                y_cv_test = [self.y_train[i] for i in test_index]

                feature_selector = SelectKBest(chi2, k=kParameter)

                x_cv_train = feature_selector.fit_transform(x_cv_train, y_cv_train)
                x_cv_test = feature_selector.transform(x_cv_test)

                clf = GaussianNB(priors=None)
                clf.fit(x_cv_train, y_cv_train)
                trainAccuracy.append(clf.score(x_cv_train, y_cv_train))
                validationAccuracy.append(clf.score(x_cv_test, y_cv_test))

            self.resTrainAccuracy.append(np.mean(trainAccuracy) * 100)
            self.resValidAccuracy.append(np.mean(validationAccuracy) * 100)
            self.resTrainError.append(stats.sem(trainAccuracy) * 100)
            self.resValidError.append(stats.sem(validationAccuracy) * 100)

            # þarf að ná í besta

    def getScore(self):
        pass
        #feature_selector = SelectKBest(chi2, k=2)
        #X_train_trans = feature_selector.fit_transform(X_train_std, y_train)
        #X_test_trans = feature_selector.transform(X_test_std)

        #clf = GaussianNB(priors=None)
        #clf.fit(X_train_trans, y_train)
        #clf.score(X_test_trans, y_test)