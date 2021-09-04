# eXplAIners
Steps and Explorations in Interpretable / Explainable AI

## Table of Contents
- [Feature Manipulations](#feature-manipulations)
- [Individual Conditional Expectations (ICE)](#individual-conditional-expectations)
- [Partial Dependence Plots (PDP)](#partial-dependence-plots)
- [Shapley Values](#shapley-values)
- [Local Interpretable Model-Agnostic Explanations (LIME)](#local-interpretable-model-agnostic-explanations)

The trained machine learning model will be referred to as the **underlying model**.

### Traditional Machine Learning Methods (Classification and Regression)
#### Feature Manipulations
The underlying model is run multiple times with different feature combinations, to get an estimate of the contribution of each feature.

A [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (from `sklearn.ensemble`) is used as the underlying model for experiments on direct feature manipulations.

##### Feature Importance
This uses the [`feature_importances_`](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) attribute of the ensemble methods in Scikit Learn which is calculated by the _Mean Decrease in Impurity (MDI)_.

<img src="https://render.githubusercontent.com/render/math?math=%24ni_j%20%3D%20w_jC_j%20-%20w_%7Bleft(j)%7DC_%7Bleft(j)%7D%20-%20w_%7Bright(j)%7DC_%7Bright(j)%7D%24">  where

<img src="https://render.githubusercontent.com/render/math?math=%24ni_j%20%3D%20%24"> The importance of node <img src="https://render.githubusercontent.com/render/math?math=%24j%24">,

<img src="https://render.githubusercontent.com/render/math?math=%24w_j%24"> = The weight of samples reaching node <img src="https://render.githubusercontent.com/render/math?math=%24j%24">,

<img src="https://render.githubusercontent.com/render/math?math=%24C_j%24"> = The impurity of node <img src="https://render.githubusercontent.com/render/math?math=%24j%24">,

<img src="https://render.githubusercontent.com/render/math?math=%24left(j)%24"> = The child node from left split on node <img src="https://render.githubusercontent.com/render/math?math=%24j%24">,

<img src="https://render.githubusercontent.com/render/math?math=%24right(j)%24"> = The child node from right split on node <img src="https://render.githubusercontent.com/render/math?math=%24j%24">,

![Feature Importances](assets/Classification/FeatureManipulations/FeatureImportances.png)

##### Feature Permutations
Here, for each feature in <img src="https://render.githubusercontent.com/render/math?math=%24(X_1%2C%20X_2%2C%20...%2C%20X_n)%24"> the feature <img src="https://render.githubusercontent.com/render/math?math=%24(X_i)%24"> is **permuted** (values are shuffled randomly), the model is retrained and the mean drop in accuracy (MDA) is measured to ascertain the importance of the feature. This was also performed on the [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
![Permutation Feature Importances](assets/Classification/FeatureManipulations/PermutedFeatureImportances.png)

##### Feature Deletions
Here, for each feature in <img src="https://render.githubusercontent.com/render/math?math=%24(X_1%2C%20X_2%2C%20...%2C%20X_n)%24"> the feature <img src="https://render.githubusercontent.com/render/math?math=%24(X_i)%24"> is **completely deleted**, the model is retrained and the mean drop in accuracy (MDA) is measured to ascertain the importance of the feature. This was done on the [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
![Deletion Feature Importances](assets/Classification/FeatureManipulations/DeletionFeatureImportances.png)


#### Individual Conditional Expectations
Individual Conditional Expectation (ICE) are a local per-instance method really useful in revealing feature interactions. 
They display one line per instance that shows how the instance’s prediction changes when a feature changes.

Formally, in ICE plots, for each instance in <img src="https://render.githubusercontent.com/render/math?math=%24%7B(x_S%5E%7B(i)%7D%2C%20x_C%5E%7B(i)%7D)%7D_%7Bi%3D1%7D%5EN%24"> the curve <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D_S%5E%7B(i)%7D%24"> is plotted against <img src="https://render.githubusercontent.com/render/math?math=%24(x_S%5E%7B(i)%7D)%24"> while <img src="https://render.githubusercontent.com/render/math?math=%24(x_C%5E%7B(i)%7D)%24"> remains fixed.

The <img src="https://render.githubusercontent.com/render/math?math=%24x_S%24"> are the feature vectors for which the ICE must be plotted and <img src="https://render.githubusercontent.com/render/math?math=%24x_C%24"> are the other features used in the underlying machine learning model <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D%24">

In these examples, the underlying model <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D%24"> used is a [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Examples from the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) are shown below.

![ICE Plot for Density](assets/Regression/ICE/ICEPlotFeaturedensity.png)

![ICE Plot Acidity vs Sulphates](assets/Regression/ICE/ICEPlotfixedacidityInteractionsulphates.png)

#### Partial Dependence Plots
Partial dependence plots (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model.
A partial dependence plot can show whether the relationship between the target and a feature is linear, monotonic or more complex.

For regression, the partial dependence function is:

<img width="368" alt="LaTeX" src="https://user-images.githubusercontent.com/13381361/132103266-2b2f132a-6501-4b49-a8da-0b6321ffc38e.png">

Again, <img src="https://render.githubusercontent.com/render/math?math=%24x_S%24"> are the feature vectors for which the ICE must be plotted and <img src="https://render.githubusercontent.com/render/math?math=%24x_C%24"> are the other features used in the underlying machine learning model <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D%24"> and the set <img src="https://render.githubusercontent.com/render/math?math=%24S%24"> which is usually small and consists only of one or two features.

The partial function <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D_%7Bx_S%7D%24"> is calculated as follows:

<img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D_%7Bx_S%7D(x_S)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%5Chat%7Bf%7D(x_S%2C%20x_C%5E%7B(i)%7D)%24">

The partial function tells us for given value(s) of features <img src="https://render.githubusercontent.com/render/math?math=%24S%24"> what the average marginal effect on the prediction is.

The underlying model <img src="https://render.githubusercontent.com/render/math?math=%24%5Chat%7Bf%7D%24"> used here is a shallow (1 hidden layer) Neural Network, an [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) with around 25 neurons in the hidden layer and the dataset used is the [`BankNotes Authentication Dataset`](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

![PDP of Entropy](assets/PDPPlotWholeEntropy.png)
![PDP of Skew](assets/PDPPlotWholeSkew.png)

#### Shapley Values
This idea comes from game theory and gives a theoretical estimate of feature prediction as compared to the above methods which were empirical and also gives importance to the sequence of features introduced.

The contribution of feature <img src="https://render.githubusercontent.com/render/math?math=%24i%24"> given the value function or **underlying model** <img src="https://render.githubusercontent.com/render/math?math=%24v%24"> is given as follows:

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cphi_i(v)%20%3D%20%5Csum_%7BS%20%5Csubseteq%20(N%20%5Cbackslash%20%5C%7Bi%5C%7D)%7D%5Cfrac%7B%7CS%7C!(%7CN%7C%20-%20%7CS%7C%20-%201)!%7D%7B%7CN%7C!%7D%20(v(S%5Ccup%7B%5C%7Bi%5C%7D%7D)%20-%20v(S))%24">

where <img src="https://render.githubusercontent.com/render/math?math=%24S%24"> is a subset of the feature set <img src="https://render.githubusercontent.com/render/math?math=%24N%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24v(S)%24"> gives the total model contribution of the subset <img src="https://render.githubusercontent.com/render/math?math=%24S%24">.

The underlying model <img src="https://render.githubusercontent.com/render/math?math=%24v%24"> used for this experiment is a [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) on the [Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and the [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) datasets.

<h5><ins>Diabetes Dataset</ins></h5>

![Shapley Values for Diabetes](assets/Classification/SHAP/Class_1/ShapleyValues.png)

<h5><ins>Wine Quality Dataset</ins></h5>

![Shapley Values on Wine Quality](assets/Regression/SHAP/ShapleyValues.png)
#### Local Interpretable Model-Agnostic Explanations
Local Interpretable Model-Agnostic Explanation (LIME) 
is a black-box **model agnostic** technique, which means it is independent of the underlying model used. It is however, **local** in nature, and generates an approximation and explanation for each example/instance of data. The explainer tries to perturb model inputs which are more interpretable to humans and then tries to generate a linear approximation _locally_ in the neighbourhood of the prediction.

In general, the overall objective function is given by

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cxi(x)%20%3D%20argmin_%7Bg%20%5Cin%20G%7D%5Cmathcal%7BL%7D(f%2C%20g%2C%20%5Cpi_x)%20%2B%20%5COmega(g)%24">

where <img src="https://render.githubusercontent.com/render/math?math=%24g(x)%24"> is the explainer function/model,

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cpi_x%24"> defines the locality/neighbourhood,

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cmathcal%7BL%7D%24"> defines the deviation or loss (or unfaithfulness) from the predictions of the actual model <img src="https://render.githubusercontent.com/render/math?math=%24f%24">,

<img src="https://render.githubusercontent.com/render/math?math=%24G%24"> is the class/family of explainable functions.

These are better illustrated by examples:
(Examples on the [Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and the [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) datasets)


Probability of being Diabetic: 0.56

Prediction: Diabetic

![LIME Explanation Instance 34](assets/Classification/LIME/LIMEexplainInstance34.png)


Predicted Wine Quality: 5.8 / 10

![LIME Explanation Instance 186](assets/Regression/LIME/LIMEexplainInstance186.png)

### Deep Learning Methods
Coming Soon!
