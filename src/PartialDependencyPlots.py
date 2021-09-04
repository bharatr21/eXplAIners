import numpy as np
import pandas as pd

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


def partial_dependence_plot(mlp, X_train, y_train, X_test):
    for idx, name in enumerate(list(X_test), 1):
        df = X_test.copy()
        mlp.fit(X_train.values, y_train)
        df.sort_values(by=name, inplace=True)
        for j in list(X_test):
            if j != name:
                df[j] = df[j].mean()
        print(df)
        y_pred = mlp.predict_proba(df.values)[:, 0]
        plt.figure(idx)
        plt.title('Partial Dependence Plot')
        plt.xlabel(name)
        plt.ylabel('Authenticity of Banknote')
        plt.plot(df[name], y_pred)
        plt.grid()
        plt.savefig('../PartialDependencyPlots/PDPAveraged_{}.png'.format(name))
        plt.show()

notes = pd.read_csv('../data/banknote.csv')
notes
X = notes.drop(columns=['Class'])
y = notes['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.015, random_state=42)

# Run the experiment and generate the PDPs (using a shallow Neural Network as the underlying Model)
mlp = MLPClassifier(hidden_layer_sizes=(25,), alpha=0.001)
mlp.fit(X_train.values, y_train)
y_pred = mlp.predict(X_test.values)
print("Accuracy: %f" %(accuracy_score(y_pred, y_test)))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred, y_test))

partial_dependence_plot(mlp, X_train, y_train, X_test)
