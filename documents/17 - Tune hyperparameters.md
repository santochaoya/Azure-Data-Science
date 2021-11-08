This document will introduce to tune hyperparamer in Azure ML. To achieve this, we can through an experiment that consists of a *hyperdrive* run, which initiates a child run for each hyperparameter combination to be tested. Each child run uses a training script with parameterized hyperparameter values to train a model, and logs the target performance metric.

* Define a hyperparameter search space
* Configure hyperparameter sampling
* Select an early-termination policy
* Run a hyperparameter tuning experiment



# Define a search space

*Search space* is the set of tried hyperparameter values during hyperparameter tuning. The range of possible value can be chosen depends on the type of hyperparameters.

## Discrete hyperpdarameters

The possible hyperparameters can be defined as a set of discrete values. Use ```choice``` from a list of explicit values, which we can define as a Python ```list```, a ```range```, a arbitrary set. Or any following discrete distributions:

* qnormal
* quniform
* qlognormal
* Qloguniform

## Continuous hyperparmeters

* normal
* uniform
* lognormal
* Loguniform



## Define a search space

```python
from azureml.train.hyperdrive import choice, normal

param_space = {'--batch_size': choice(16, 32, 64),
               '--learning_rate': normal(10, 3)}
```



# Configuring sampling

The specific values used in hyperparameter tuning run depedens on the type of *sampling* used.

## Grid sampling

Only used on all hyperparameters are **discrete**, and is used to try every possible combination of parameters in the search space.

```python
from azureml.train.hyperdrive import GridParameterSampling, choice

param_space = {'--batch_size': choice(16, 32, 64),
               '--learning_rate': choice(0.01, 0.1, 1.0)}

param_sampling = GridParameterSampling(param_space)
```



## Random sampling

Can be used on a **mix of discrete and continuous values**, will randomly slelect a value for each hyperparameter.

```python
from azureml.train.hyperdrive import RandomParameterSampling, choice, normal

param_space = {'--batch_size': choice(16, 32, 64),
               '--learning_rate': normal(10, 3)}

param_sampling = RandomParameterSampling(param_space)
```



## Bayesian sampling

Based on the Bayesian optimization algorithm, which tries to **select parameter combinations that will result in improved performance from the previous selection**.

```python
from azureml.train.hyperdrive import BayesianParameterSampling, choice, normal

param_space = {'--batch_size': choice(16, 32, 64),
               '--learning_rate': uniform(0.05, 0.1)}

param_sampling = BayesianParameterSampling(param_space)
```



# Configuring early termination

Set an early termination policy that abandons runs that are unlikely to produce a better result than previously completed runs. The evaluation is based on each time the target performance metric is logged. * *

* ```evaluation_interval``` the target performance metric is logged
* ```delay_evaluation``` avoid evaluating the policy until a minimum number of ilerations have been completed.



## Bandit policy

Stop a run when it underperforms a specified margin from the best run so far.

```python
from azureml.train.hyperdriver import BanditPolicy

early_termination_policy = BanditPolicy(slack_amount=0.2,
                                        evaluation_interval=1,
                                        delay_evaluation=5)
```

* ```slack_amount``` the specific margin
* ```evaluation_interval``` the specific margin will compare the target metric to the best performing run after the number of interval
* ```delay_evaluation``` policy apply after the first five iteration

The policy will aplly after the first five iteration, then it will compare the target metric to the best performaing run from the previous interval(here is 1) iteration. if worse than 0.2, it will stop.



## Median stopping policy

Stops when the target metric is worse than the mdeian of the running averages for all runs.

```python
from azureml.train.hyperdrive import MdeianStoppingPolicy

early_termination_policy = MedianStoppingPolicy(evaluation_interval=1,
                                                delay_evaluation=5)
```



## Truncation selection policy

When specify a ```truncation_percentage```, the policy cancels the lowest performing X% of runs at each evaluation interval.

```python
from azureml.train.hyperdriver import TruncationSelectionPolicy

early_termination_policy = TruncationSelectionPolicy(truncation_percentage=10,
                                        						 evaluation_interval=1,
                                        						 delay_evaluation=5)
```



# Run a hyperparameter tuning experiment

## Create a training script

* Include an argument for each hyperparameter
* Log the target performance metric. (For selecting the best performce model)

Using ```--regularization``` and ```Accuracy```:

```python
import argparse
import joblib
import azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get reqularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', deault=0.01)
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run contxt
run = Run.get_context()

# Load the training dataset
data = run.input_datasets['training_data'].to_pandas_dataframe()

# Seperate features and labels, and split for training/validation
X = data[['feature1', 'feature2']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a logistic regression model with the reg hyperparameter
model = LogsiticRegression(C=1/reg, solver='liblinear').fit(X_train, y_train)

# Calculate and log accuracy
y_pred = model.predict(X_test)
acc = np.average(y_pred==y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()
```

## configuring and runnning a hyperparameter experiment

```python
from azureml.core import Experiment
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

# Assume ws, script_config and param_sampling are already define

hyperdrive = HyperDriveConfig(run_config=script_config,
                              hyperparameter_sampling=param_sampling,
                              policy=None,
                              primary_metric_name='Accuracy',
                              primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                              max_total_runs=6,
                              max_concurrent_runs=4)

experiment = Experiment(workspace=ws,
                        name='hyperdrive_training')
hyperdrive_run = experiment.submit(config=hyperdrive)
```



## Monitoring and reviewing hyperdrive runs

* Monitoring

  * Retrieve

  The experiment will initiate a child run for each hyperparameter combination to be tried, and you can retrieve the logged metrics these runs using the following:

  ```python
  for child_run in run.get_children():
  		print(child_run.id, child_run.get_metrics())
  ```

  * List all runs in descending order of performance

  ```python
  for child_run in hyperdrive_run.get_children_sorted_by_primary_metric():
  		print(child_run)
  ```

  * Retrieve the best performance:

  ```python
  best_run = hyperdrive_run.get_best_run_by_primary_metric()
  ```





