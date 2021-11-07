This documents is to introduce how to create and use environment and compute targes.



# Environment

## Create

### From files

Use a Conda or pip specification file to define packages, stored a file named **conda.yml**

```bash
named: py_env
dependencies:
	- numpy
	- pandas
	- scikit-learn
	- pip:
		- azureml-defaults
```

Create environment from saved **conda.yml**

```python
from azureml.core import Environment

env = Environment.from_conda_specification(name='training_environment',
                                           file_path='./conda.yml')
```



### From existing Conda Environment

If already have an existing Conda Environment defined on the workspace

```python
from azureml.core import Environment

env = Environment.from_existing_conda_environment(name='traning_environment',
                                                  conda_environment='py_env')
```



### From a specifying packages

Define environment from a Conda and pip packages in ```CondaDependencies```objects

```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environmemnt('training_environment')
deps = CondaDependencies.create(conda_packages=['scikit-learn', 'pandas', 'numpy'],
                                pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages
```



## Configuring environment containers

Environment for experiment script defaultly created in containers.

> Change the default environment created in container:
>
> Use a ```DockerConfiguration``` with a ```use_docker``` to ```False```

*ScriptRunConfig*

```python
from azureml.core import Experiment, ScriptRunConfig
from azureml.core.runconfig import DockerConfiguration

docker_config = DockerConfiguration(use_docker=True)

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                environment=env,
                                docker_runtime_config=docker_config)
```

Azure ML uses a library of base images for containers. If we created a custom container and registered it in a container registry, it will be override the default base images and use the container we created by modifying the attributes of the environment's **docker** property.

```python
env.docker.base_image='my-base-image'
env.docker.base_image_registry='myregistry.azurecr.io/myimage'
```

Alternatively, we also can have an image created on-demand based on the base image and addtional settings in a dockerfile.

```python
env.docker.base_image = None
env.docker.base_dockerfile = './Dockerfile'
```

By default, Azure ML handles Python paths and package dependencies. If our image already includes them, we can override these by setting ```python.user_managed_dependencies``` to ```True```

```python
env.python.user_managed_dependencies = True
env.python.interpreter_path = '/opt/miniconda/bin/python'
```



## Register

```python
env.register(workspace=ws)
```

View after registering

```python
from azureml.core import Environment

env_names = Environment.list(workspace=ws)
for env_name in env_names:
		print(env_name)
```



## Retrieve

Using the ```get```method to retrieve a registered environment, then assign it to a ```ScriptRunConfig```

*ScriptRunConfig*

```python
from azureml.core import Environment, ScritRunConfig

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                environment=training_env)
```



# Compute Target

Computer targets are physical or virtual computers on which experiments are run.

## Types of compute

* **Local compute** - based on your physical workstation or a virtual machine. Suitable for development and testing with low to moderate volumes of data

* **Compute clusters** - multi-node clusters of Virtual Machines taht automatically scale up or down to meet demand.

  Suitable for handing large volumes of data or use parallel processing to distribute the workload and reduce the time it take to run.

* **Attached compute** - Already use the Azure-based compute environment, can attached it to Azure ML workspace



## Create with SDK

### Compute Cluster

Use ```AmlCompute``` class, create a 4 nodes which is based on the STANDARD_DS12_v2 virtual machine image.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarge, AmlCompute

ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuraion(vm_size='SDANARD_DS11_V2',
                                                      min_nodes=0,
                                                      max_nodes=4,
                                                      vm_priority='dedicated')

# Create the compute
aml_cluster = ComputeTraget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)
```



### Attaching an unmanaged compute target

*Unmanaged compute target* is that defined and managed outside of the Azure ML workspace. For example, an Azure virtual machine or an Azure Databricks cluster.

Using ```ComputeTarget.attach()``` method to attach the existing compute based on its target specific configuration settings.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute

ws = Workspace.from_config()

compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db.workspace_name,
                                                   access_token=db_access_token)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)
```



### Checking for an existing compute target

Using ```ComputeTargetException```

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = 'aml-cluster'

# Check if the compute target exist
try:
		aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
  	# If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_v2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
    
aml_cluster.wait_for_completion(show_output=True)
```



### Use Compute targets

Use a particular compute target by specifying it in the appropriate parameter for an experiment run configuration or estimator. For example, use the compute target named *aml_cluster*.

```python
from azureml.core import Environment, ScriptRunConfig

compute_name = 'aml-cluster'
training_env = Environment.get(workspacke=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                environment=training_env,
                                compute_target=training_cluster)
```


