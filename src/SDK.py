from shared_code.PrinttingTemplate import *

from azureml.core import Workspace, Experiment
import pandas as pd


# -----------------------------------------------------------------------------------------------------------------------------

# Start Printing
starter('Azure Machine Learning with Python SDK')

# Workspace
section_label('Connect to Workspace')

ws = Workspace.from_config()

# Work with Workspace
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)

section_ending()

# -----------------------------------------------------------------------------------------------------------------------------

# Experiment
section_label('Connect to Experiment')

# Connect to an experiment
experiment = Experiment(workspace=ws, name='SDK-exercise')

# Start the experiment
run = experiment.start_logging()

# Exercuse the experiment
data = pd.read_csv('../data/wine.csv')
row_count = (len(data))

# Log the row count
run.log('observations', row_count)

# End the experiment
run.complete()

section_ending()