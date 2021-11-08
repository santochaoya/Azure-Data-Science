**Learning objectives**

* Interpret *global* and *local* feature importance
* Use an explainer
* Create model explanations in a training experiment
* Visualize model explanations



# Feature Importance

## Global feature importance

*Global feature importance* quantifies the relative importance of each feature in the test dataset as a whole. It measures the influence of each feature in the datataset to the whole predictions.



## Local feature importance

*Local feature importance* features the influence of each feature value for a specific individual prediction.



# Use explainer

## Create an explainer

Install the ```azureml-interpret``` package to create an explainer to interpret a local model. The type of explainer, including:

* ```MimicExplainer``` - an explainer that creates a **global surrogate model** which approximates the trained model and can generate explanations. *Must have the same architecture as the trained model(tree-based or linear)*

* ```TabularExplainer``` - act as a wrapper around various **SHAP explainer algorithms**, automatically chose the most approrpriate.
* ```PFIExplainer``` - a **Permutation Feature Importance** explainer that analyzes feature importance by **shuffling feature values** and measuring the impace on prediction performance.

```python
# MinicExplainer
from interpret.ext.blackbox import MiniExplainer
from interpret.ext.glassbox import DecisionTreeExplainableModel

mim_explainer = MimicExplainer(model=loan_model,
                               initialization_examples=X_test,
                               explainable_model = DecisionTreeExplainableModel,
                               features=['loan_amount', 'income', 'age', 'marital_status'],
                               classes=['reject', 'approve'])

# TabularExplainer
from interpret.ext.blackbox import TabularExplainer

tab_explianer = TabularExplainer(model=loan_model,
                                 initialization_examples=X_test,
                                 features=['loan_amount', 'income', 'age', 'marital_status'],
                                 classes=['reject', 'approve'])

# PFIExplainer
from interpret.ext.blackbox import PFIExplainer

tab_explianer = PFIExplainer(model=loan_model,
                             features=['loan_amount', 'income', 'age', 'marital_status'],
                             classes=['reject', 'approve'])
```



## Explaining global feature importance

```python
# MimicExplainer
global_mim_explanation = mim_explainer.explain_global(X_train)
global_mim_feature_importance = global_mim_explanation.get_feature_importance_dict()


# TabularExplainer
global_tab_explanation = tab_explainer.explain_global(X_train)
global_tab_feature_importance = global_tab_explanation.get_feature_importance_dict()


# PFIExplainer
global_pfi_explanation = pfi_explainer.explain_global(X_train, y_train)
global_pfi_feature_importance = global_pfi_explanation.get_feature_importance_dict()
```



## Explaining local feature importance

```python
# MimicExplainer
local_mim_explanation = mim_explainer.explain_local(X_test[0:5])
local_mim_features = local_mim_explanation.get_ranked_local_names()
local_mim_importance = local_mim_explanation.get_ranked_local_values()


# TabularExplainer
local_tab_explanation = tab_explainer.explain_local(X_test[0:5])
local_tab_features = local_tab_explanation.get_ranked_local_names()
local_tab_importance = local_tab_explanation.get_ranked_local_values()
```



# Create explanations

## Create an explanation in the experiment script

```python
from azureml.core.run import Run
from azureml.contrib.interpret.explanation_client import ExplanationClient
from interpret.ext.blackbox import TabularExplainer

run = Run.get_context()

# code to train model here

# Get explanation
explainer= TabularExplainer(model, X_train, features=features, classes-labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient,from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

# Complete the run
run.complete()
```



## Viewing the explanation

```python
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient

client = ExplanationClient.from_run_id(workspace=ws,
                                       experiment_name=experiment.experiment_name,
                                       run_id=run.id)
explanation = client.download_model_explanation()
feature_importances = explanation.get_feature_importance_dict()
```



# Visualizing explanations

Using Azure ML Studio to get multiple visualizations

* **Explanations** tab : global feature importance
* **Summary Importance** tab: the distribution of individual importance values for each feature across the test dataset
* Select an individual value to show the local feature importance.