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
        
```

