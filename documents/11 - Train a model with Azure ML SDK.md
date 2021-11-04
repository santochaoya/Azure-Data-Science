This documents is to use a ``ScriptRunConfig`` to run a reusable, parameterized training script as a Azure ML experiment.



# Script to train a model

## Save the model as script for an experiment

Save a trained model to the outputs of an experiment.



## Running the model script as an experiment

A script which need to run as an experiment must set for

* **environment**
  * create the environment
  * ensure the required packages installed
* **config**
* **Submit**