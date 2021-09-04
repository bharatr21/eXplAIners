import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pdpbox import pdp, info_plots
from pycebox.ice import ice, ice_plot
import shap
import lime
import eli5
from eli5.sklearn import PermutationImportance

def load_dataset(file: str, sep=','):
    if file.endswith('.csv'):
        df = pd.read_csv(file, sep=sep)
    elif file.endswith('.xls') or file.endswith('.xlsx'):
        df = pd.read_excel(file)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return df, X, y


df, X, y = load_dataset('../data/diabetes.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
def callee(X):
    return model.predict_proba(X)[:, 1]

def underlying_model(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def metrics(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    fscore = f1_score(y_test, y_pred)
    print("Confusion Matrix\n {}".format(confusion_matrix(y_test, y_pred)))
    print("Model Accuracy: {}".format(acc))
    print("Model Precision: {}".format(prec))
    print("Model Recall: {}".format(rec))
    print("Model F-Score: {}".format(fscore))
    return acc


def shapley_explainer(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)
    expl = shap.TreeExplainer(model)
    shap_values = expl.shap_values(X)
    # Needs JS in the IPython Notebook (Can run only in iPython/Jupyter Notebook)
    # shap.initjs()
    # shap.force_plot(expl.expected_value, shap_values[0,:], X.iloc[0,:], show=False)
    # plt.savefig('../Plots/AllPlots/ShapleyForcePlotInstance1.png')
    # shap.force_plot(expl.expected_value, shap_values, X, show=False)
    # plt.savefig('../Plots/AllPlots/ShapleyForcePlotAvg.png')
    for feat in X.columns:
        plt.figure()
        shap.dependence_plot(feat, shap_values[0], X, show=False)
        plt.savefig('../Plots/Classification/SHAP/ShapleyFeature_{}.png'.format(feat), bbox_inches='tight')
    
    plt.figure(100)
    shap.summary_plot(shap_values[0], X, show=False,  plot_type='bar')
    plt.savefig('../Plots/Classification/SHAP/ShapleyFeatImps.png', bbox_inches='tight')
    plt.savefig('../Plots/Classification/SHAP/ShapleyFeatImps.pdf', bbox_inches='tight')
    plt.figure(101)
    shap.summary_plot(shap_values[0], X, show=False)
    plt.savefig('../Plots/Classification/SHAP/ShapleyValues.png', bbox_inches='tight')
    plt.savefig('../Plots/Classification/SHAP/ShapleyValues.pdf', bbox_inches='tight')

def lime_explainer(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, inst_no: int, mode='classification'):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)

    expl = lime.lime_tabular.LimeTabularExplainer(class_names=['Non-Diabetic', 'Diabetic'], feature_names=X_train.columns, training_data=X_train.values, verbose=True,mode=mode)
    exp = expl.explain_instance(X_test.values[inst_no, :], model.predict_proba, num_features=len(X_test.columns))
    with open('../Plots/Classification/LIME/LIME_Instance_{}.txt'.format(inst_no), 'w+') as f:
        f.write('Instance no: {}'.format(inst_no))
        f.write('\n')
        f.write('Intercept: {}'.format(exp.intercept[1]) if exp.intercept[1] else 'Intercept: {}'.format(exp.intercept[0]))
        f.write('\n')
        f.write('Probability (Diabetic): {}'.format(model.predict_proba(X_test.values[inst_no, :].reshape(1, -1))[:, 1]))
        f.write('\n')
        f.write('Model Accuracy: {}'.format(np.mean(accuracy_score(y_test, model.predict(X_test)))))
        f.write('\n')
        f.write('\n')
        f.write('Explaining Instance: {}\n---------------------\n'.format(inst_no))
        f.write('\n')
        content = '\n'.join(['{}\t\t{}'.format(i[0], i[1]) for i in exp.as_list()])
        f.write(content)
        f.write('\n')
    
    fig = exp.as_pyplot_figure()
    fig.savefig('../Plots/Classification/LIME/LIME_Instance_{}.png'.format(inst_no), bbox_inches='tight')
    fig.savefig('../Plots/Classification/LIME/LIME_Instance_{}.pdf'.format(inst_no), bbox_inches='tight')
    exp.save_to_file('../Plots/Classification/LIME/LIME_Instance_{}.html'.format(inst_no))

def partial_dependence_plot(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)

    for feat in X_train.columns:
        info_plots.target_plot(df=df, feature=feat, feature_name=feat, target=y.name, show_percentile=True)
        plt.savefig('../Plots/Classification/PDP/TargetPlotFeature_{}.png'.format(feat), bbox_inches='tight')
        plt.savefig('../Plots/Classification/PDP/TargetPlotFeature_{}.pdf'.format(feat), bbox_inches='tight')
        info_plots.actual_plot(model, X_train, feature=feat, feature_name=feat, show_percentile=True, predict_kwds={})
        plt.savefig('../Plots/Classification/PDP/ActualPlotFeature_{}.png'.format(feat), bbox_inches='tight')
        plt.savefig('../Plots/Classification/PDP/ActualPlotFeature_{}.pdf'.format(feat), bbox_inches='tight')
        pdplotter = pdp.pdp_isolate(model, dataset=X_train, model_features=X_train.columns, feature=feat, num_grid_points=20)
        pdp.pdp_plot(pdplotter, feat)
        plt.savefig('../Plots/Classification/PDP/PartialDepPlotFeature_{}.png'.format(feat), bbox_inches='tight')
        plt.savefig('../Plots/Classification/PDP/PartialDepPlotFeature_{}.pdf'.format(feat), bbox_inches='tight')

def individual_conditional_expectation(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)

    for feat in X_test.columns:
        ice_df = ice(X_test, feat, callee, num_grid_points=100)
        fig = plt.figure(figsize=(16, 6))
        ice_ax = plt.gca()

        ice_plot(ice_df, frac_to_plot=0.1, c='k', alpha=0.25, ax=ice_ax, plot_pdp=True, pdp_kwargs={'c': 'cyan', 'path_effects': [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]})
        ice_ax.set_xlabel(feat)
        ice_ax.set_ylabel(y_test.name)
        ice_ax.set_title('ICE curve for {}'.format(feat))
        plt.savefig('../Plots/Classification/ICE/ICEPlotFeature_{}'.format(feat), bbox_inches='tight')
    
    for feat in X_test.columns:
        for feat2 in X_test.columns:
            if feat != feat2:
                fig = plt.figure(figsize=(16, 6))
                ice_ax = plt.gca()

                ice_plot(ice_df, frac_to_plot=0.1, color_by=feat2, cmap='PuOr', ax=ice_ax, plot_points=True, point_kwargs={'c':'k', 'alpha':0.75})
                ice_ax.set_xlabel(feat)
                ice_ax.set_ylabel(y_test.name)
                ice_ax.set_title('ICE Curves of {} coloured by {}'.format(feat, feat2))
                plt.savefig('../Plots/Classification/ICE/ICEPlot_{}_Interaction_{}'.format(feat, feat2), bbox_inches='tight')

def feature_manipulations(df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame):
    model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)
    
    feats = [x[0] for x in sorted(zip(X_test.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)]
    vals = [x[1] for x in sorted(zip(X_test.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)]

    plt.barh(y=feats, width=vals)
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.xlabel('Normalized Feature Importance')    
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    plt.savefig('../Plots/Classification/FeatureManipulations/FeatureImportances.png', bbox_inches='tight')
    plt.savefig('../Plots/Classification/FeatureManipulations/FeatureImportances.pdf', bbox_inches='tight')

    perm = PermutationImportance(model.fit(X_train, y_train)).fit(X_test, y_test)

    pfeats = [x[0] for x in sorted(zip(X_test.columns, perm.feature_importances_), key=lambda x: x[1], reverse=True)]
    pvals = [x[1] for x in sorted(zip(X_test.columns, perm.feature_importances_), key=lambda x: x[1], reverse=True)]

    plt.figure()
    plt.barh(y=pfeats, width=pvals, color='orange')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.xlabel('Normalized Permutation Feature Importance')    
    plt.ylabel('Permuted Feature')
    plt.title('Permutation Feature Importance Plot')
    plt.savefig('../Plots/Classification/FeatureManipulations/PermutedFeatureImportances.png', bbox_inches='tight')
    plt.savefig('../Plots/Classification/FeatureManipulations/PermutedFeatureImportances.pdf', bbox_inches='tight')

    featdelsacc = []
    acc = metrics(df, X, y)
    for feat in X.columns:
        df, X, y = load_dataset('../data/diabetes.csv')
        del X[feat]
        model, X_train, X_test, y_train, y_test = underlying_model(df, X, y)
        y_pred = model.predict(X_test)
        deltacc = acc - accuracy_score(y_test, y_pred)
        featdelsacc.append((feat, deltacc))
    
    print(featdelsacc)
    dfeats = [x[0] for x in sorted(featdelsacc, key=lambda x: x[1], reverse=True)]
    dvals = [x[1] for x in sorted(featdelsacc, key=lambda x: x[1], reverse=True)]

    plt.figure()
    plt.barh(y=dfeats, width=dvals, color='red')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.xlabel('Normalized Feature Deletion Importance')    
    plt.ylabel('Deleted Feature')
    plt.title('Deletion Feature Importance Plot')
    plt.savefig('../Plots/Classification/FeatureManipulations/DeletionFeatureImportances.png', bbox_inches='tight')
    plt.savefig('../Plots/Classification/FeatureManipulations/DeletionFeatureImportances.pdf', bbox_inches='tight')



df, X, y = load_dataset('../data/diabetes.csv')
# Uncomment one by one to run the various explainers on the features
# shapley_explainer(df, X, y)
# lime_explainer(df, X, y, np.random.randint(1, 152))
# partial_dependence_plot(df, X, y)
# individual_conditional_expectation(df, X, y)
# feature_manipulations(df, X, y)
# metrics(df, X, y)