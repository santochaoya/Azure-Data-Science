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



# Azure Machine Learning SDK

## Installation

```bash
pip install azureml-sdk
```

and also can install the external package with

```bash
pip install azureml-sdk azureml-widgets
```



## Connect to a Workspace