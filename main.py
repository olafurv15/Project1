import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from Classifiers.NaiveBayes import NaiveBayes
from Classifiers.Knn import Knn

# Changing time format to seconds
def stringToSeconds(time):
    time = time.split(":")
    return int((time[0])) * 60 + int(time[1])

def preProcessingData(data, targetString):
    # Drop unnecessary columns
    NbaShots = data.drop(
        ['GAME_ID', 'MATCHUP', 'W', 'FINAL_MARGIN', 'PTS_TYPE', 'CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID', 'FGM',
         'PTS', 'player_name', 'player_id'], axis=1)

    # Drop all null values
    NbaShots = NbaShots.dropna()
    NbaShots = NbaShots.reset_index(drop=True)

    # Changing Location column to a numeric values
    NbaShots.loc[NbaShots['LOCATION'] == 'H', 'LOCATION'] = 1
    NbaShots.loc[NbaShots['LOCATION'] == 'A', 'LOCATION'] = 0

    # Changing GAME_CLOCK format
    NbaShots['GAME_CLOCK'] = NbaShots['GAME_CLOCK'].apply(stringToSeconds)

    # Dropping all rows which have negative value.
    NbaShots = NbaShots[NbaShots >= 0]
    NbaShots = NbaShots.dropna()
    NbaShots = NbaShots.reset_index(drop=True)

    # SHOT_RESULT is our target and therefore dropped
    data = NbaShots.drop(['SHOT_RESULT'], axis=1)
    target = NbaShots['SHOT_RESULT']

    #data = data.dropna()
    #data = data.reset_index(drop=True)

    # Dropping Target
    #target = data[targetString]

    #data = data.drop([targetString], axis=1)
    # þarf að tjekka betur
    #data = data._get_numeric_data()

    #data = data.select_dtypes(exclude=['object'])

    #data = data.reset_index(drop=True)

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
    NbaShots = pd.read_csv('shot_logs.csv')
    targetString = 'SHOT_RESULT'

    data, target = preProcessingData(NbaShots, targetString)

    # splitting data.
    X_train, X_test, y_train, y_test = splittingDataAndNormalize(data, target)

    #knn = Knn(X_train, y_train, X_test, y_test)

    #result = knn.getScore()
    # print(result)

    #tree = DecisionTree(X_train, y_train, X_test, y_test)
    bayes = NaiveBayes(X_train, y_train, X_test, y_test)

    bayes.printBestScoreAndParam()





