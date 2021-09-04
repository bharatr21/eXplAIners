## Experiments

### Direct Feature Manipulation and Surrogate Models
----------------------
1.1) Feature Deletion
1.2) Feature Permutation
1.3) Feature Importance or Variable Importance (Not available for all models)
1.4) Surrogate Models like Logistic Regression, Decision Tree which are "inherently" interpretable

#### Global Model Agnostic Explainers
----------------------
2.1) PDP (Partial Dependence Plots)
2.2) ICE (Centered, Differential or Individual Conditional Expectation)
2.3) BETA (Black Box Explanations through Transparent Approximations) and other global methods, if any found

#### Local Model Agnostic Explainers
----------------------
3.1) LIME
3.2) aLIME (Anchor has technical difficulties, will add in soon)
3.3) Shapley Values (Use `shap` package) since it is local to an instance

