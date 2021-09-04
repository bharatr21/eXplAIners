import numpy as np
import pandas as pd

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

def surrogateDT(X_train, y_train, X_test, y_pred):
    X = pd.concat((X_train, X_test), axis=0, ignore_index=True)
    y = np.concatenate((y_train, y_pred), axis=0)
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X, y)
    y_surr = dt.predict(X_test.values)
    print("Fidelity Score: {}".format(accuracy_score(y_surr, y_pred)))
    print("Accuracy Score: {}".format(accuracy_score(y_surr, y_test)))
    print("Fidelity Report")
    print(classification_report(y_surr, y_pred))
    print("Accuracy Report")
    print(classification_report(y_surr, y_test))

notes = pd.read_csv('../data/banknote.csv')
X = notes.drop(columns=['Class'])
y = notes['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(25,), alpha=0.001)
mlp.fit(X_train.values, y_train)
y_pred = mlp.predict(X_test.values)
y_prob = mlp.predict_proba(X_test.values)[:, 0]
print("Accuracy: %f" %(accuracy_score(y_pred, y_test)))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred, y_test))

# partial_dependence_plot(mlp, X_test) #Partial Dependence Plot of various features wrt Authenticity. Needs testsize <= 0.03
surrogateDT(X_train, y_train, X_test, y_pred) #Requires larger test sizes >= 0.8