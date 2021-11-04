from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies


# Create a python environment
sklearn_env = Environment('sklearn-env')

# Ensure the required packages are installed
packages = CondaDependencies.create(pip_packages=['azureml-defaults','scikit-learn'])
sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='training',
                                script='training.py',
                                environment=sklearn_env)

# Submit the experiment
ws = Workspace.from_config()

experiment_name = 'SDK-exercise'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
run.complete()

