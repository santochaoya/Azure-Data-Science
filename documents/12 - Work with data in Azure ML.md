This documents is to introduce how to:

* create and use a datastore with SDK
* create and use a dataset with SDK



# Datastores

Datasotres are the abstractions for cloud data sources. They encapsulate the information required to connect to data sources.

## Types of Datastore

There are data types that Azure Machine Learning supported:

* Azure Storage(blob and file containers)
* Azure Data Lake stores
* Azure SQL Database
* Azure Databricks file system(DBFS)



## Built-in Datastores

Azure Machine Learning used two built-in datastore (**Azure Storage blob containter** and **Azure Storage file container** ). Or you can use your own data sources.



## Use datastores

### Register a datastore to workspace

* Register it using the Azure Machine Learning studio by graphical interface

* SDK

  ```python
  from azureml.core import Workspace, Datastore
  
  ws = Workspace.from_config()
  
  # Register a new datastore
  blob_ws = Datastore.register_azure_blob_containter(workspace=ws,
                                                     datastore_name='blob_data',
                                                     container_name='data_container',
                                                     account_name='az_store_acct',
                                                     account_key='12345abcde789...')
  ```



### Manaing datastore

* **View**

  ```python
  for ds_name in ws.datastore:
  		print(ds_name)
  ```

* **Get a reference**

  ```python
  blob_store = Datastore.get(ws, datastore_name='blob_data')
  ```

* **Built-in default datastore: workspaceblobstore**

  ```python
  default_store = ws.get_default_datastre()
  ```

* **Change the default datastore**

  ```python
  ws.set_default_datastore('blob_data')
  ```



# Datasets

## Type of dataset

* **Tabular** 

  The data is read from the dataset as tabular. When prefer to use a structured tabular data. Panda dataframe.

* **File**

  The dataset present a list of file paths.



## Create and Register Dataset

* Azure Machine Learning studio
* SDK



### Create and register TABULAR datasets

Using ```from_delimited_files``` method of ```Dataset.Tabular``` class to create dataset from individual files or multiple file paths. The paths can include wildcards(for example, */files/.csv*)

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
						 (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(paths=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```

This example include files from:

* The **current_data.csv** under folder **data/files**
* All .csv files under folder **data/files/archive**

After creating the dataset, register to the workspace with the name **csv_table**.



### Create and Register FILE datasets

Using ```from_files``` method of ```Dataset.File```

```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/image/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
```



## Retrieving a registered dataset

After registering the dataset, we can retrieving it by using any of the following techniques:

* The ```datasets``` dictionary attribute of a ```Workspace``` object.
* The ```get_by_name``` or ```get_by_id``` method of the ```Dataset``` class

For example:

```python
from azureml.core import Workspace, Dataset

ws = Workspace.from_config()

# DS1 - from the workspace dataset collection
ds1 = ws.datasets['csv_table']

# DS2 - by name from the dataset class
ds2 = Dataset.get_by_name(ws, 'img_files')
```



## Versioning

The control of version for a dataset enable us to track the historical versions of datasets that we used in the experiment, and reporduce those experiments with data in the same state.



### Create and register a version

Create a new version by registering it with a specifiying the **create_new_version** property when register the dataset.

```python
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
						 (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
```



### Retrieving a specific dataset version

Specifying the **version** parameter in the ```get_by_name```method.

```python
img_ds = Dataset.get_by_names(workspace=ws, name='img_files', version=2)
```



## Work with TABULAR datasets

Read data directly from a tabular by converting it into Pandas or Spark DataFrame.

* Pandas

  ```python
  df = tab_ds.to_pandas_dataframe()
  ```

* Spark dataframe

  ```python
  df = tab-ds.to_spark_dataframe()
  ```



### Pass a TABULAR to an experiment script

When access a dataset in an exeperiment script, we have to pass the dataset to the script with the following two techniques:



#### Use a script ARGUMENT for a tabular dataset

Using an argument to pass the tabular dataset to experiment script. The argument will received by the script is the unique ID for the dataset in the workspacek

*ScriptRunConfig*: Config the dataset to get an experiment script

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-default',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env)
```



*Script:* Access the dataset with unique id

```python
from azureml.core import Run, Dataset

parser.add_argument('--ds', type=str, dest='dataset_id')
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()
```



#### Use a NAMED INPUT for a tabular dataset

Using a named input to pass the tabular to an experiment script. The ```as_named_input``` method of the dataset to specify a name of the dataset.

*ScriptRunConfig*

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-default',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                arguments=['--ds', tab_ds.as_named_input('my_dataset')],
                                environment=env)
```

*Script*

```python
from azureml.core import Run

parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args()

run = Run.get_context()
dataset = run.input_datasets['my_dataset']
data = dataset.to_pandas_dataframe()
```



## Work with file datasets

Use the ```to_path()```return a list of file paths

```python
for file_path in file_ds.to_path():
		print(file_path)
```



### Pass a file dataset to an experiment script

#### Argument

Unlike the tabular, we must specify a mode for the file dataset argument, which can be ```as_download``` or ```as_mount```. This provides that the script can read the file ini the dataset. 

* ```as_download``` will copy the file to a temporary location on the compute.
* ```as_mount``` will stream the files directly from the source when the storage space on the experiment is not enough.

​	*ScriptRunConfig*

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-default',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                arguments=['--ds', file_ds.as_download()],
                                environment=env)
```

*Script*

```python
from azureml.core import Run
import glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

imgs = glob.glob(args.ds_ref + "/*.jpg")
```



#### Named Input

* Specify the name: ```as_named_input```
* Retrieve the dataset: ```input_datasets```

*ScriptRunConfig*:

```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-default',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='outputs',
                                script='script.py',
                                arguments=['--ds', 
                                           file_ds.as_named_input('my_ds').as_download()],
                                environment=env)
```

*Script*

```python
from azureml.core import Run
imoprt glob

parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

dataset = run.input_datasets['my_ds']
data = glob.glob(dataset + '/*.jpg')
```

