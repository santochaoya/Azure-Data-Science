This document will introduce :

* Publish batch inference pipeline for a trained model
* Use a batch inference pipeline to generate predictions



# Create

## 1. Register a model (Same as document 15)

Register a model from a local file in the Azure ML workspace.

```python
from azureml.core import Model

classification_model = Model.register(workspace=ws,
                                      model_name='classification_model',
                                      model_path='model.pkl',
                                      description='A classification model')
```

or, if reference to the ```Run```

```python
run.register_model(model_name='classification_model',
                   model_path='outputs/model.pkl', # run output path
                   description='A classification model')
```



## 2. Create a scoring script

The same as the **entry script** in [document 15](/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/documents/15 Deploy real-time ML services.md), use ```init()``` and ```run()``` to load model and predict new values. The difference is: the input value of ```run()``` function is ```mini_batch``` instead of ```raw_data```

* ```init()```: Called when the pipeline is initialized.
* ```run(mini_batch)```: Called for **each batch of data to be processed**

The same as the **entry script** for a ML service, ```init()``` used to load the model registered, ```run``` to generate predictions from each batch of data.

```python
import os
import numpy as np
import azureml.core import Model
import joblib

def init():
  	"""Runs when the pipeline step is initialized"""
    global model
    
    # load the model
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)
    
def run(mini_batch):
  	"""Runs for each batch"""
    results_list = []
    
    # process each file in the batch
    for f in mini_batch:
      	# Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-demensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        result_list.append("{}:{}".format(os.path.basename(f), prediction[0]))
    return result_list
        
```



## 3. Create a pipeline

Use ```ParallelRunStep``` to read batches of files from a **File** dataset and write the processing output to a ```OutputFileDatasetConfig```. Set the ```output_action``` to ```append_now``` will ensure that all instances of the step being run in parallel will collate their results to a single output file named *parallel_run_step.txt*.

```python
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline

# Get the batch dataset for input
batch_data_set = ws.datasets['batch-data']

# Set the output location
default_ds = ws.get_default_datastore()
output_dir = OutputFileDatasetConfig(name='inferences')

# Define the parallel run step configuration
parallel_run_config = ParallelRunConfig(source_directory='batch_scripts',
                                        entry_script='batch_scoring_script.py',
                                        mini_batch_size=5,
                                        error_threshold=10,
                                        output_action='append_now',
                                        environment=batch_env,
                                        compute_target=aml_cluster,
                                        node_count=4)

# Create the parallel run step
parallelrun_step = ParallelRunStep(name='batch-score',
                                   parallel_run_config=parallel_run_config,
                                   inputs=[batch_data_set.as_named_input('batch_data')],
                                   output=output_dir,
                                   arguments=[],
                                   allow_reuse=True)

# Create the pipeline
pipeline = Pipeline(workspacke=ws, steps=[parallelrun_step])
```



## 4. Run the pipeline and retrieve the step output

```python
from zureml.core import Experiment

# Run the pipelne as an experiment
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Get the outputs from the first step (only from here!!)
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='results')

# Find the parallel_run_step.txt file
for root, dirs, files in os.walk('results'):
  	for file in files:
      	if file.endswith('parallel_run_step.txt'):
        		result_file = os.path.join(root, file)

# Load and display the results
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ['File', 'Prediction']
print(df)
```



# Publish a batch inference pipeline

## Publish as a REST service

```python
published_pipeline = pipeline_run.publish_pipeline(name='Batch_Prediction_Pipeline',
                                                   description='Batch pipeline',
                                                   version='1.0')
rest_endpoint = published_pipeline.endpoint
```

Use the service endpoint to initiate a batch inferencing job

```python
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={'ExperimentName': 'Batch_Prediction'})
run_id = response.json()['id']
```

Schedual the published pipeline to have it run automatically

```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

weekly = ScheduleRecurrence(frequency='Week', interval=1)
pipeline_schedule = Schedule.create(ws, name='Weekly Predictions',
                                    description='batch inferencing',
                                    pipeline_id=published_pipeline.id,
                                    experiment_name='Batch_Prediction',
                                    recurrence=weekly)
```

