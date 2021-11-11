This is a summarized document of creating a model service with Azure ML Python SDK

**Learning Objectives**:

* Create or load Azure ML workspace
* Register MLflow model with Azure ML and build container image deployment
* Deploy model to Azure Container Instance(ACI)
* Deploy model to Azure Kubernates Service(AKS)
* Ttest model deployments
* Update model deployment in AKS



# Create or load Azure ML workspace

Using ```azureml.core.Workspace.create()``` to load a workspace of a specified name or create if not exists.

```python
from azureml.core import Workspace


workspace_name = "<workspace-name>"
workspace_location = "<workspace-location>"
resource_group = "<resource-group>"
subscription_id = "<subscription-id>"

workspace = Workspace.create(workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             exist_ok=True)
```



# Train the model and build the container

## Train the Diabetes Model

Using the ```diabetes``` dataset in scikit-learn and predicts the progression metric.

```0python
import os
import warnings
import sys
from random import random
import pandas as pd
import numpy as np
from itertools immport cycle
from sklearn.metrics import mean_squred_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
# import mlflow
import mlflow
import mlflow.sklearn


# Load diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

def train_diabetas(data, in_alpha, in_ll_ratio):
  	# Evaluate metrics
    def eval_metrics(actual, pred):
      	rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absoluate_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        return rmse, mae, r2
    
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    
```

