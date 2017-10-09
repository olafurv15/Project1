import numpy as np

from sklearn.model_selection import KFold
from scipy import stats
from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2

class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        #                                   var i 1000
        self.min_splitValues = np.arange(2, 100, 10)

        self.resTrainAccuracy4 = []
        self.resValidAccuracy4 = []
        self.resTrainError4 = []
        self.resValidError4 = []

        self.kf = KFold(n_splits=2)

        self.getSplitParameterWithCrossValidation()
        #
        self.kBest = np.arange(1, 10, 1)
        self.resTrainAccuracy5 = []
        self.resValidAccuracy5 = []
        self.resTrainError5 = []
        self.resValidError5 = []

    def getSplitParameterWithCrossValidation(self):
        for splitParameter in self.min_splitValues:
            trainAccuracy = []
            validationAccuracy = []

            for train_index, test_index in self.kf.split(self.X_train):
                x_cv_train = [self.X_train[i] for i in train_index]
                y_cv_train = [self.y_train[i] for i in train_index]
                x_cv_test = [self.X_train[i] for i in test_index]
                y_cv_test = [self.y_train[i] for i in test_index]

                # train decision tree
                dt = tree.DecisionTreeClassifier(min_samples_split=splitParameter)
                dt.fit(x_cv_train, y_cv_train)

                param = dt.get_params()

                # add the results to a lost
                trainAccuracy.append(dt.score(x_cv_train, y_cv_train))
                validationAccuracy.append(dt.score(x_cv_test, y_cv_test))

            self.resTrainAccuracy4.append(np.mean(trainAccuracy) * 100)
            self.resValidAccuracy4.append(np.mean(validationAccuracy) * 100)
            self.resTrainError4.append(stats.sem(trainAccuracy) * 100)
            self.resValidError4.append(stats.sem(validationAccuracy) * 100)

    def getKBestWithCrossValidation(self):
        for kParameter in self.kBest:
            trainAccuracy = []
            validationAccuracy = []

            for train_index, test_index in self.kf.split(self.X_train):
                # print("splitting data")
                x_cv_train = [self.X_train[i] for i in train_index]
                y_cv_train = [self.y_train[i] for i in train_index]
                x_cv_test = [self.X_train[i] for i in test_index]
                y_cv_test = [self.y_train[i] for i in test_index]

                # select features
                k = kParameter
                feature_selector = SelectKBest(chi2, k=kParameter)
                x_cv_train = feature_selector.fit_transform(x_cv_train, y_cv_train)
                x_cv_test = feature_selector.transform(x_cv_test)

                # train decision tree
                dt = tree.DecisionTreeClassifier()
                dt.fit(x_cv_train, y_cv_train)

                # add the results to a list
                trainAccuracy.append(dt.score(x_cv_train, y_cv_train))
                validationAccuracy.append(dt.score(x_cv_test, y_cv_test))

            # Calculate the means, stderror for each cross validation and add to  our result arrays
            self.resTrainAccuracy5.append(np.mean(trainAccuracy) * 100)
            self.resValidAccuracy5.append(np.mean(validationAccuracy) * 100)
            self.resTrainError5.append(stats.sem(trainAccuracy) * 100)
            self.resValidError5.append(stats.sem(validationAccuracy) * 100)

    def getScore(self):
        pass
        # select features
        #feature_selector = SelectKBest(chi2, k=8)
        #x_cv_train = feature_selector.fit_transform(X_train, y_train)
        #x_cv_test = feature_selector.transform(X_test)

        # train decision tree
        #dt = tree.DecisionTreeClassifier(min_samples_split=530)
        #dt.fit(x_cv_train, y_train)
        #dt.score(x_cv_test, y_test)