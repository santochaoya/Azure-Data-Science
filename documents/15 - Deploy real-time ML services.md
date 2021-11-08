This document is about with deploying real-time machine learning services:

* Deploy a model as a real-time inferencing services.
* Consume a real-time inferencing service.
* Troubleshoot service deployment



# Introduction

The trained model will deploy as a part of a servide that enables applications to request inmediate.

In Azure ML, using **Azure Kubernetes Services(AKS)** which is a containerized platform to deploy a model as a service.



# Deploy a model as a real-time service

## 1. Register a trained model

Using the ```register``` method of the ```Model``` object to register a model **from a local file.**

```python
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                                      model_name='classification_model',
                                      model_path='model.pkl', # a local path
                                      description='A classification model')
```

or, if reference to the ```Run```

```python
run.register_model(model_name='classification_model',
                   model_path='outputs/model.pkl', # run output path
                   description='A classification model')
```



## 2. Define an inference configuration

The model deployed as a service will consist of:

* A script to load the model and return prediction for submitted dat
* An environment in which the script will be run

Must define a script and environment for the service



### Create the entry script

Create the entry script for the service as a Python file must include **tow functions**:

* ```init()``` : Called when the **service is initialized**
* ```run(raw_data)```: Called when **new data is submitted** to the service

Typically, use the ```init()``` load the model from the model registry, use the ```run``` function to generate the precitions from the input data.

```python
import json
import joblib
import numpy as np
import os

# Called when the service is loaded
def init():
		global model
    # Get the path to registered model file and load
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
    
# Called when a request is received
def run(raw_data):
  	# Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['daata'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()
```

For example, then save the script about as *score.py* in a folder named *service_files*



### Create an environment

Create an environment in which to run the entry script.

```python
from azureml.core import Environment

service_env = Enrvironment(name='service-env')
python_packages = ['scikit-learn', 'numpy'] # whatever packages your entry script uses
for package in python_packages:
  	service_env.python.conda_dependencies.add_pip_packages(package)
```



### Combine the script and environment

```python
from azureml.core.model import InferenceConfig

classifier_inference_config = InferenceConfig(source_direcgory='service_files',
                                              entry_script='score.py',
                                              environment=service_env)
```



## 3. Define a deployment configuration

If deploying to an AKS cluster, must create the cluster and a compute target before deploying:

```python
from azureml.core.compute import ConputeTarget, AksCompute

cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(loacation='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)
```

Define the deployment configuration, which sets the target-specific compute specification for the containerized deployment:

```python
from azureml.core.webservice import AksWebservice

classifier_deploy_config = AksWebservice.deploy_configuration(cpu_core=1,
                                                              memory_gb=1)
```



## 4. Deploy the model

```python
from azureml.core.model import Model

model = ws.models['classification_model']
service = Model.deploy(workspace=ws,
                       name='classifier-service',
                       models=[model],
                       inference_config=classifier_inference_config,
                       deployment_config=classifier_deploy_config,
                       deployment_target=production_cluster)
service.wait_for_completion(show_output=True)
```

For ACI or local services, set the ```deployment_target```  to a specific value or ```None```



# Consume a real-time inferencing service

After deploying a real-time service, you can consume it from client applications to predict labels for new data cases.



## Use SDK

Use SDK for tesing by calling a **web service** through the ```run``` method of a ```WebService``` object. Typically, we can send data to the ```run``` method in JSON format with the following structure:

```json
{
  "data":[
    	[0.1, 2.3, 4.1, 2.0], // 1st case
    	[0.2, 1.8, 3.9, 2.1], // 2nd case
    	...
  ]
}
```

The response from the ```run``` method is a JSON collection with a prediction form each case.

For example, call a service and displays the response:

```python
import json

# Get new dataset convert to a JSON document
x_new = [[0.1, 2.3, 4.1, 2.0], 
				[0.2, 1.8, 3.9, 2.1]]
json_data = json.dumps({'data': x_new})

# Call the web service, passing the input data
response = service.run(input_data=json_data)

# Get the prediction
predictions = json.loads(response)

# Print the predicted class for each case
for i in range(len(x_new)):
  	print(x_new[i], predicions[i])
```



## Use a REST endpoint

If the client application doesn't inclue the Azure ML SDK, we can determine the endpoint of a deployed a service in Azure ML studio. or by retrieving the ```scoring_uri``` property of the ```WebService``` in SDK:

```python
endpoint = service.scoring_uri
print(endpoint)
```

Then when known the endpoint, can use the POST request with JSON data to call the service.

```python
import requests
import json

# Get new dataset convert to a JSON document
x_new = [[0.1, 2.3, 4.1, 2.0], 
				[0.2, 1.8, 3.9, 2.1]]
json_data = json.dumps({'data': x_new})

# Set the content type in the request headers
request_headers = {'Content-Type': 'application/json'}

# Call the service
response = requests.post(url=enpoint,
                         data=json_data,
                         headers=request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case
for i in range(len(x_new)):
  	print(x_new[i], predictions[i])
```



### Authentication

Restrict access to the service by applying authentication.  There are two kinds of authentication:

* **Key**: Specifying the key
* **Token**: a JSON Web Token (JWT)

By default, authentication is disabled for ACI services, and set to key-based authentication for AKS services (for which primary and secondary keys are automatically generated). You can optionally configure an AKS service to use token-based authentication (which is not supported for ACI services).



#### Key-based

Retrieve the keys for a service:

```python
primary_key, secondary_key = service.get_keys()
```

#### Token-based

To make an authenticated call to the service's REST endpoint, you must include the **key or token** in the request header like this:

```python
import requests
import json

# Get new dataset convert to a JSON document
x_new = [[0.1, 2.3, 4.1, 2.0], 
				[0.2, 1.8, 3.9, 2.1]]
json_data = json.dumps({'data': x_new})

# Set the content type in the request headers
request_headers = {'Content-Type': 'application/json',
									 'Authorization': 'Bearer ' + key_or_token}

# Call the service
response = requests.post(url=enpoint,
                         data=json_data,
                         headers=request_headers)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case
for i in range(len(x_new)):
  	print(x_new[i], predictions[i])
```



# Troubleshoot

## Check the service state

In an initial trouble shotting step

```python
from azureml.core.webservice import AskWebservice

# Get the deployed service
service = AksWebservice(name='classifier-service', workspace=ws)

# Check its state
print(service.state)
```



## Review service logs

Review it

```python
print(service.get_logs())
```

The logs include detailed information about the provisioning of the service, and the requests it has processed. They can often provide an insight into the cause of unexpected errors.



## Deploy to a local container

Deploy a service as a container in a local Docker instance

```python
from azureml.core.webservice import LocalWebService

deployment_config = LocalWebService.deploy_configuration(port=8888)
service = Model.deploy(ws, 'test-svc', [model], inference_config, deployment_config)
```

test

```python
print(service.run(input_data = json_data))
```

You can then troubleshoot runtime issues by making changes to the scoring file that is referenced in the inference configuration, and reloading the service without redeploying it

```python
service.reload()
print(service.run(input=json_data))
```

