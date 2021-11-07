This document is about creating, publish and scheduling an Azure ML Pipeline.



# Introduction

In Azure ML, a *pipeline* is a workflow of machine learning tasks in which each task is implemented as a **step**.

A pipeline can be executed to run as an experiment, can be published as a REST endpoint. 



# Pipeline steps

Common kinds of steps include:

* ```PythonScriptStep```: Runs a specified Python Script.
* ```DataTransferStep```: Uses Azure Data Factory to copy data between data stores.
* ```DatabricksStep```: Runs a notebook, script, or a compiled JAR on a databricks cluster.
* ```AdlaStep```: Runs a U-SQL job in Azure Data Lake Analytics
* ```ParallelRunStep```: Runs a Python Script as a distributed task on multiple compute nodes.

Before creating a pipeline, we need to define each step depends on the step type and then create a pipeline that includes the steps.

*PythonScriptStep*

```python
from azureml.pipeline steps import PythonScriptStep

# Step to run a Python script
step1 = PythonScriptStep(name='prepare data',
                         source_directory='outputs',
                         script_name='data_prep.py',
                         compute_target='aml-cluster')

# Step to train a model
step2 = PythonScriptStep(name='train model',
                         source_directory='outputs',
                         script_name='train_model.py',
                         compute_target='aml-cluster')
```

Assign steps to a pipeline, and run it as an experiment:

```python
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

# Construct the pipeline
train_pipeline = Pipeline(workspace=ws, steps=[step1, step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name='training-pipeline')
pipeline_run = experiment.submit(train_pipeline)
```



# Pass data between pipeline steps

When a pipeline includes a step that depends on the output of a preceding step, we need to use a python script to preprocess some data, which will used in a subsequent step to train a model.

## The ```OutputFileDatasetConfig``` Object

The ```OutputFileDatasetConfig``` object is a special kind of dataset:

* References a location in a datastore for interim storage of data
* Creates a data dependency between pipeline steps

It can be treated as an intermediary store for data that must be passed from a step to a subsequent step

<img src="/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/Images/pipeline 1.png" alt="pipeline 1" style="zoom: 33%;" />

### Inputs and Outputs of ```OutputFileDatasetConfig``` step

1. Define a named ```OutputFileDatasetConfig``` object that **references a location** in a datastore. If there is no explicit datastore, the default datastore is used.
2. Pass the ```OutputFileDatasetConfig``` as a **script argument** in step that run scripts.
3. Include code in those scripts to write to the ```OutputFileDatasetConfig``` **argument as an output or read it as an input.**

```python
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Get a datset for the initial data
raw_ds = Dataset.get_by_name(ws, 'raw_dataset')

# Define a PipelineData to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = OutputFileDatasetConfig('prepped')

# Step to run a Python Script
step1 = PythonScriptStep(name='prepare data',
                         source_directory='outputs',
                         script_name='data_prep.py',
                         compute_target='aml-cluster',
                         # Script arguments include PipelineData
                         arguments=['--raw-ds', raw_ds.as_named_input('raw_data'),
                                    '--out_folder', prepped_data])

# Step to run an estimator
step2 = PythonScriptStep(name='train model',
                         source_directory='scripts',
                         script_name='train_model.py',
                         compute_target='aml-cluster',
                         # Pass as script argument
                         arguments=['--training-data', prepped_data.as_input()])
```

or obtain a reference to the ```OutputFileDatasetConfig``` from the script argument, and use it like a local folder.

```python
# code in the data_prep.py
from azureml.core import Run
import argparse
import os

run = Run.get_context()

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw_ds', type=str, dest='raw_data_id')
parser.add_argument('--out_folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

# Get input dataset as dataframe
raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# code to prep data
prepped_df = raw_df[['col1', 'col2']]

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'prepped_data.csv')
prepped_df.to_csv(output_path)
```



# Reuse pipeline steps

To reduce the cost of time for running pipelines with multiple long-running steps, we can use caching and reuse features. 

## Managing step output reuse

Use ```allow_reuse``` parameter to control reuse for an individual step

```python
step1 = PythonScriptStep(name='prepare data',
                         source_directory='outputs',
                         script_name='data_prep.py',
                         compute_target='aml-cluster',
                         runconfig=run_config,
                         inputs=[raw_ds.as_named_input('raw_data')],
                         outputs=[prepped_data],
                         arguments=['--folder', prepped_data],
                         # Disable step reuse
                         allow_reuse=False)
```



## Forcing all steps to run

By setting the ```regenerate_outputs``` parameter when submitting the pieline experiment

```python
pipeline_run = experiment.submit(train_pipeline, regenerate_outputs=True)
```



# Publish Pipelines

## Publish

```python
published_pipeline = pipeline.publish(name='training_pipeline',
                                      description='Model training pipeline',
                                      version='1.0')
```



## Review

Review it after publishing a pipeline

```python
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
```



## Use a published pipeline

Use a published pipeline by an HTTP requests to its REST endpoint, with an authorization header and a token for a service principal with permission to run the pipeline.

```python
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline"})
run_id = response.json()['id']
print(run_id)
```



# Pipeline parameter

## Define parameters

Create a ```PipelineParameter``` for each parameter and specify them in one step.

For example, we use a parameter for a regularization rate by an estimator:

```python
from azureml.pipeline.core.graph import PipelineParameter

reg_param = PipelineParameter(name='reg_rate', default_value=0.01)
...
step2 = PythonScriptStep(name='train model',
                         source_directory='outputs',
                         script_name='data_prep.py',
                         compute_target='aml-cluster',
                         # Pass parameter as script argument
                         arguments=['--in_folder', prepped_data,
                                    '--reg', reg_param],
                         inputs=[prepped_data])
```

> **Notes**:
>
> Parameters must be defined before publishing a pipeline.



## Running with parameter

Pass parameter values in the JSON payload for the REST interface:

```python
repsonse = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={'ExperimentName': 'run_training_pipeline',
                               'ParameterAssignments': {'reg_rate': 0.01}})
```



# Schedule pipelines

After publishing a pipeline, we can initiate it on demand throught its REST endpoint, or have the pipelines run automatically based on a periodic schedule or in response to data updates.

## Scheduling a pipeline for periodic intervals

Define a ```ScheduleRecurrence``` to determine the run **frequency**, and use it to create a **Schedule**.

```python
from azureml.pipeline core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training',
                                    description='trains model every day',
                                    pipeline_id=published_pipeline.id,
                                    experiment_name='Training_Pipeline',
                                    recurrence=daily)
```



## Triggering a pipeline run on data changes

To schedule a pipeline to run whenever data changes, we must create the ```Schedule``` monitors a path on datastore.

```python
from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline_id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='data/training')
```

