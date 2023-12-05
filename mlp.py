# Angelo Pilotto Rodrigues Alves - nro USP: 12542647

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from numpy import average
import csv

def normalize_dataset(db, y_min, y_max):
    n_col = len(db[0])
    maxs = [0]*n_col
    mins = [0]*n_col
    found_min_max = [False]*n_col
    for line in db:
        for i in range(n_col):
            if not found_min_max[i]:
                maxs[i] = max(db, key=lambda line: line[i])[i]
                mins[i] = min(db, key=lambda line: line[i])[i]
                found_min_max[i] = True
            line[i] = normalize_val(line[i], mins[i], maxs[i], y_min, y_max)
    return [mins, maxs]

def normalize_val(x, x_min, x_max, y_min, y_max):
    return ((x-x_min)/(x_max-x_min))*(y_max-y_min)+y_min

def preprocess_data(csv_file_path):
    X = []
    y = []
    with open(csv_file_path) as csv_file_read:
        csv_reader = csv.reader(csv_file_read)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            row[1] = 1 if row[1] == 'Masculino' else 0
            row[2] = int(row[2])
            row[3] = float(row[3].replace(',', '.'))
            row[4] = int(row[4])
            X.append(row[2:])
            y.append(row[1])

    [mins, maxs] = normalize_dataset(X, 0, 1)

    return X, y, mins, maxs

def preprocess_data_no_hand_size(csv_file_path):
    X = []
    y = []
    with open(csv_file_path) as csv_file_read:
        csv_reader = csv.reader(csv_file_read)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            row[1] = 1 if row[1] == 'Masculino' else 0
            row[2] = int(row[2])
            row[4] = int(row[4])
            X.append([row[2], row[4]])
            y.append(row[1])

    [mins, maxs] = normalize_dataset(X, 0, 1)

    return X, y, mins, maxs

X, y, mins, maxs = preprocess_data('data.csv')

scores = []
averages = []
clf = MLPClassifier(hidden_layer_sizes=[4], learning_rate_init=0.03, max_iter=10000)

for _ in range(10):
    for __ in range(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    averages.append(average(scores))
    scores = []

X, y, mins, maxs = preprocess_data_no_hand_size('data.csv')

no_hand_scores = []
no_hand_averages = []
clf = MLPClassifier(hidden_layer_sizes=[4,4], learning_rate_init=0.03, max_iter=10000)

for _ in range(10):
    for __ in range(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf.fit(X_train, y_train)
        no_hand_scores.append(clf.score(X_test, y_test))
    no_hand_averages.append(average(no_hand_scores))
    no_hand_scores = []

print('Com tamanho da mão:', average(averages))
print('Sem tamanho da mão:', average(no_hand_averages))
