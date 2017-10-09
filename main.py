import pandas as pd
from Knn import Knn
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from scipy import stats
from sklearn import tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import sys

def preProcessingData(data, targetString):

    data = data.dropna()
    data = data.reset_index(drop=True)

    # Dropping Target
    target = data[targetString]

    data = data.drop([targetString], axis=1)
    # þarf að tjekka betur
    #data = data._get_numeric_data()

    data = data.select_dtypes(exclude=['object'])

    data = data.reset_index(drop=True)

    return data, target

def splittingDataAndNormalize(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.8)

    # Normalize the data
    std_scale = preprocessing.Normalizer().fit(X_train)
    X_train_std = std_scale.transform(X_train)
    X_test_std = std_scale.transform(X_test)

    #X_train_std = X_train_std.as_matrix()
    y_train = y_train.as_matrix()

    return X_train_std, X_test_std, y_train, y_test

if __name__ == "__main__":
    #data = pd.read_csv(sys.argv[1])
    #targetString = sys.argv[2]
    data = pd.read_csv('shot_logs.csv')
    targetString = 'SHOT_RESULT'

    data, target = preProcessingData(data, targetString)

    # splitting data.
    X_train, X_test, y_train, y_test = splittingDataAndNormalize(data, target)

    #knn = Knn(X_train, y_train, X_test, y_test)

    #result = knn.getScore()
    #tree = DecisionTree(X_train, y_train, X_test, y_test)
    bayes = NaiveBayes(X_train, y_train, X_test, y_test)


    #print(result)





