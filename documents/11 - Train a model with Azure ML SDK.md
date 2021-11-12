This document is to use a ``ScriptRunConfig`` to run a reusable, parameterized training script as an Azure ML experiment.



# Script to train a model

## Save the model as a script for an experiment

Save a trained model to the outputs of an experiment.

```python
from azureml.core import Run
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
data = pd.read_csv('data.csv')
X, y = data[['Feature1', 'Feature2']].values, data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver='liblinear').fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
acc = np.average(y_pred == y_test)
run.log('Accuracy', acc)

# Save the model
os.makedirs('outputs', exist_ok=True)
joblib.dumps(value=model, name='outputs/model.pkl')

# Complete the experiment
run.compelete()

```



Save the script as ```training.py``` in the folder ```training```.

## Running the model script as an experiment

A script that need to run as an experiment must be set for

```python
# Ensure packaged installed
packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
																		pip_packages=['azureml-defaults'])
sklearn_env.python.conda_denpendencies = packagesfrom azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

# Create the Python environment for the experiment
sklearn_env = Environment('sklearn-env')

# Ensure packaged installed
packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
																		pip_packages=['azureml-defaults'])
sklearn_env.python.conda_denpendencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='training',
																script='training.py',
                               	environment=sklearn_env)

# Submit the experiment
experiment = Experiment(workspace=ws, name='SDK-exercise')
run = experiment.submit(config=script_config)
run.wait_for_completion()
```



# Using Script Parameters

## Setting the hyperparameters using arguments

When training the model, we always use the hyperparameters to find the model with the best performance. Using the library such as ```argparse``` to read the arguments passed to the scripe and assign them to the variables.

For example, we use ```--reg-raate``` used to set the regularization rate hyperparameter for the logistic regression algorithm used to train model.

Add a section of setting regularization hyperparameter before training the model.

```python
import argparse

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate
```



## Passing arguments to an experiment script

In the experiment script, passing the arguments to **ScriptRunConfig**. The argument value containing a list of comma-separated arguments.

```python
# Create a script config
script_config = ScriptRunConfig(source_directory='training',
                               	script='training.py',
                                arguments=['--reg-rate', 0.1],
                               	environment=sklearn_env)
```



# Registering models

Use the **Run** objects to retrieve its outputs, and the training model.



## Retrieving model files

* ```get_file_names```: list the the file generated.
* ```download_file``` or ```download_files```: download the files to the local folder

```python
# List the file names
for file in run.get_file_names():
  	print(file)

# Download a named file
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')
```



## Registering a model

Register a model can specify a name, description, tags, framework(such as Scikit-learn or PyTorch), framework version, custom properties. and other useful metadata. 

* Register a model from the local file

  ```python
  from azureml.core import Model
  
  model = Model.register(workspace=ws,
                         model_name='classification_model',
                         model_path='model.pkl', # local path
                         description='A classfication model.',
                         tags={'data-format' : 'CSV'},
                         model_framework=Model.Framework.SCIKITLEARN,
                         model_framework_version='0.20.3')
  ```



* Register from a reference 

  Get rid of ```workspace=ws```

  Replace ```model_path='model.pkl'``` to ```model_path='outputs/model.pkl'```



## View the registerd models

```python
from azureml.core import Model

for model in Model.list(ws):
		# Get the model name and automated-generated version
		print(model.name, 'version:', model.version)
```



