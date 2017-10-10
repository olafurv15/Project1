from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


class Knn:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.score = 0

        # creating list of K for KNN
        self.neighbors = list(range(1, 201))
        # take every other number
        self.neighbors = self.neighbors[::2]

        # empty list that will hold cv scores
        self.cv_scores = []

        self.crossValidation()
        self.optimal_k = self.findOptimal_k()
        self.computeScore()


    def crossValidation(self):

        # perform cross validation
        for k in self.neighbors:
            # build the model
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, self.X_train, self.y_train, cv=20, scoring='accuracy')

            # record accuracy
            self.cv_scores.append(scores.mean())

    def findOptimal_k(self):
        # changing to misclassification error
        MSE = [1 - x for x in self.cv_scores]

        # best k.
        return self.neighbors[MSE.index(min(MSE))]

    def computeScore(self):
        clf = KNeighborsClassifier(n_neighbors=self.optimal_k)
        clf.fit(self.X_train, self.y_train)
        self.score = clf.score(self.X_test, self.y_test)

    def getScore(self):
        return self.score

    def getClassifierName(self):
        return "K-nearest neighbour"

    def printBestScoreAndParam(self):
        print("\nK-Nearest Neighbor\n-------------------\n" +
              "Best N neighbors parameter: " + str(self.optimal_k) + "\n"+
              "Score: " + str(self.score) + "\n")
