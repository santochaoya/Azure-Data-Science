from shared_code.PrinttingTemplate import *

from azureml.core import Workspace, Experiment


# Connect a Workspace
ws = Workspace.from_config()


print('=========================================  Connect to Workspacke  =================================================')

for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)

print('-------------------------------------------------------------------------------------------------------------------')
print('=========================================  Connect to Experiment  =================================================')

# Connect to an experiment
experiment = Experiment(workspace=ws, name='SDK-exercise')

