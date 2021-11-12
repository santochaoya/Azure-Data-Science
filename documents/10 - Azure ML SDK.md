Besides processing the resources with Microsoft Azure Portal, use the Azure Machine Learning Python SDK to run code can accomplish these targets as well.



# Workspaces

## Machine Learning Assets

* **Compute**: for development, training, and deployment
* **Data**
* **Notebooks**: contains shared code and documentations
* **Experiments**: include run history with logged and metrics
* **Pipelines**: define multi-step processes
* **Models**

![AzureML_SDK1](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\AzureML_SDK1.png)

The Azure resources created alongside a workspace include:

- **Storage Account**: Store files used by the workspace as well as data for experiments and model training.
-  **Application Insights**: monitor predictive services in the workspace.
- **Azure Key Vault**: Manage secrets such as authentication keys and credentials used by the workspace.
- **Container Registry**: Manage containers for deployed models.



## Create a Workspace

There are several ways to create a workspace:

* In the **Microsoft Azure portal**

* Use the **Azure Machine Learning Python SDK** to run code to create a workspace

  ```python
  from azureml.core import Workspace
  
  ws = Workspace.create(name='aml-workspace',
  					subscription_id='123456-abc-123...',
  					resource_group='aml-resources',
  					create_resource_group=True,
  					location='eastus')
  					
  ```

* Use the Azure Command Line interface(CLI) with the Azure Machine Leaning CLI extension. 

  ```bash
  az ml workspace create -w 'aml-workspace' -g 'aml-resources'
  ```

* Create an **Azure Resource Manager** template



## Installation- SDK

```bash
pip install azureml-sdk
```

and also can install the external package with

```bash
pip install azureml-sdk azureml-widgets
```



## Connect to a Workspace

We can also connent a workspace from an existing configuration files in JSON format. For example, ```config.json```

```json
{
  	"subscription_id": "<your subscription id here>",
  	"resource_group": "XimeCraft_MachingLearning",
    "workspacke_name": "AzML_Regression"
}
```

Then connect the workspace in Python SDK:

```python
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, "loaded")
```

## View Azure ML resources in the workspace

```python
print('Compute Resources: \n')
for compute_name in ws.compute_targets:
  	compute = ws.compute_targets[compute_name]
    print('\t', compute.name, ':', compute.type)
```



# Experiment

## Create Experiment

A script that need to run as an experiment must be set for

### **environment**

* create the environment

  ```python
  from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
  from azureml.core.conda_dependencies import CondaDependencies
  
  # Create the Python environment for the experiment
  sklearn_env = Environment('sklearn-env')
  ```

  

* ensure the required packages installed

  ```python
  # Ensure packaged installed
  packages = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
  																		pip_packages=['azureml-defaults'])
  sklearn_env.python.conda_denpendencies = packages
  ```

  

### **config**

configuration of script, same as previous document

```python
# Create a script config
script_config = ScriptRunConfig(source_directory='training',
																script='training.py',
                               	environment=sklearn_env)
```



### **Submit**

Submit the experiment

```python
experiment = Experiment(workspace=ws, name='SDK-exercise')
run = experiment.submit(config=script_config)
run.wait_for_completion()
```





