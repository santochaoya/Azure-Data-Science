**Learning objectives**

* Use Azure ML's automated machine learning capabilities to determine the best performing algorithm
* Use automated ml to preprocess data for training
* Run an automated ml experiment



# Automated ml tasks and algorithms

## Tasks

- Classification
- Regression
- Time Series Forecasting



## Algorithms

### Classification algorithms

- Logistic Regression
- Light Gradient Boosting Machine (GBM)
- Decision Tree
- Random Forest
- Naive Bayes
- Linear Support Vector Machine (SVM)
- XGBoost
- Deep Neural Network (DNN) Classifier
- Others...

### Regression algorithms

- Linear Regression
- Light Gradient Boosting Machine (GBM)
- Decision Tree
- Random Forest
- Elastic Net
- LARS Lasso
- XGBoost
- Others...

### Forecasting algorithms

- Linear Regression
- Light Gradient Boosting Machine (GBM)
- Decision Tree
- Random Forest
- Elastic Net
- LARS Lasso
- XGBoost
- Others...

## Restrict algorithm selection

The automated machien learning will randomly select from the full range of algorithms for a specified task. We can restrict algorithm from being selected.



# Preprocessing and Featurization

## Scaling and normalization

Automated ml will apply scaling and normalization to **numeric data** automatically.



## Optional featurization

The preprocessing transformations we can choose:

* Missing value imputation
* Categorical encoding
* Drop high-cardinality features, such as IDs.
* Feature engineering
* Others....



# Automated ML experiment

## Configure an automated ml experiment

Using ```AutoMLConfig``` class

```python
from azureml.train.automl import AutoMLConfig

automl_run_config = RunConfiguration(framework='python')
automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             primary_metric='AUC_weighted',
                             compute_target=aml_compute,
                             training_data=train_dataset,
                             validation_data=test_dataset,
                             label_column_name='Label',
                             featurization='auto',
                             iterations=12,
                             max_concurrent_iterations=4)
```



## Specify data for training

We can create or select an Azure ML dataset from Azure ML studio. Or use SDK

When use SDK

* if a validation dataset isn't specified, automated Azure ML will apply cross-validation using the training data



## Specify the primary metric

This is the target performance metric. To retrieve it by using ```get_primary_metrics```

```python
from azureml.train.automl.utilities import get_primary_metrics

get_primary_metrics('classification')
```



## Submit an automated ml experiment

```python
from azureml.core.experiment import Experiment

automl_experiment = Experiment(ws, 'automl_experiment')
automl_run = automl_experiment.submit(automl_config)
```



## Retrieve the best run and model

```python
best_run, fitted_model = automl_run.get_output()
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
		metric = best_run_metrics[metric_name]
		print(metric_name, metric)
```



## Explore preprocssing steps

```python
for step_ in fitted_model.named_steps:
		print(step_)
```

