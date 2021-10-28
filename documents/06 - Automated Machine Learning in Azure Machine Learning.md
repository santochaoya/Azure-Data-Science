This document is to introduce how to setup an Azure Machine Learning and use it to train and manage a model.



# Preparation

## Azure ML workspace

1. Sign into the https://portal.azure.com/

2. Select **+ Create a resource**, search for Machine Learning, and create a new **Machine Learning ** resource with settings:

   * **Subcription**: *Your Azure Subscription (Free Trial)*

   * **Resource group**: *Create or select a resource group*
   * **Workspace name**: *Enter a unique name for your workspace*
   * **Region**: *Select the geographical region closest to you*
   * **Storage account**: *Default*
   * **Key vault**: *Default*
   * **Application insights**: **Default**
   * **Container registry**: None (*one will be created automatically the first time you deploy a model to a container*)

3. Wait for the workspace to be created. The status will be *running*.
4. On the *Overview* page, *Launch Studio*.
5. Toggle  ☰ icon to manage the resources in the workspace.



## Compute Resource

Create a compute resource to calculate the training process.



### Create compute targets

Compute targets are cloud-based resources on which you can run model training and data exploration processes.

==**Compute Resources**==:

* **Compute Instances**: Deployment workstations to process data and model, e.g:

  * **Compute name**: *enter a unique name*

  * **Virtual Machine type**: CPU

  * Virtual Machine size:

    - Choose **Select from all options**

    - Search for and select **Standard_DS11_v2**

      

* **Compute Cluster**: Scalable clusters of virtual machines for on-demand processing of experiment code, e.g. :

  * **Location**: *Select the same as your workspace. If that location is not listed, choose the one closest to you*

  * **Virtual Machine priority**: Dedicated

  * **Virtual Machine type**: CPU

  * Virtual Machine size

    :

    - Choose **Select from all options**
    - Search for and select **Standard_DS11_v2**

  * **Compute name**: *enter a unique name*

  * **Minimum number of nodes**: 0

  * **Maximum number of nodes**: 2

  * **Idle seconds before scale down**: 120

  * **Enable SSH access**: Unselected

    

* **Inference Clusters**: Deployment targets for predictive service
* **Attached Compute**: Links to existing Azure compute resources(Virtual Machines, Azure Databricks)

> After completing each module, be sure to follow the **Clean Up** instructions at the end of the module to **stop** your compute resources. **Stopping** your compute ensures your subscription won't be charged for compute resources.



## Dataset

### From web files

Create a new dataset on *Dataset* page **from web files**, using an example with the following settings:

- Basic Info:
  - **Web URL**: https://aka.ms/bike-rentals
  - **Name**: bike-rentals
  - **Dataset type**: Tabular
  - **Description**: Bicycle rental data
  - **Skip data validation**: *Do not select*
- Settings and preview:
  - **File format**: Delimited
  - **Delimiter**: Comma
  - **Encoding**: UTF-8
  - **Column headers**: Only first file has headers
  - **Skip rows**: None
  - **Dataset contains multi-line data**: *Do not select*
- Schema:
  - Include all columns other than **Path**
  - Review the automatically detected types
- Confirm details:
  - Do not profile the dataset after creation



### From Pipeline

#### Create Pipeline

1. On the **Pipeline** page, click on the default pipeline name(**Pipeline-Created-on-date**) and change it.
2. Beside the pipeline name, go to **Settings** pane (**⚙** icon ), click on **Select compute target** to select **compute cluster**

3. After transformation the data, **Submit the pipelie**

#### Explore dataset

1. We can get the **missing value** of each column. 
   * Some of them have quite a few missing values in the column. This will make them usefulness in predicting the label. So it is better to **exclude these columns**
   * Some of the columns have few missing values. So if **exclude the missing rows**, they will still be useful.
2. **Normalize the features**





# Train a model

## Create Experiment

**Experiments** are the operations you run. Create a new *Automated ML* for an example rental bake dataset which we created before, with the following settings:

==**Steps**==

* Select **+ new Azure ML run**

- Select dataset: (input dataset)
  - **Dataset**: bike-rentals
- Configure run:
  - **New experiment name**: mslearn-bike-rental
  - **Target column**: rentals (*this is the **label** the model will be trained to predict)*
  - **Select Azure ML compute cluster**: *the compute cluster you created previously*
- Select task type:
  - **Task type**: Regression *(the model will predict a numeric value)*

- Additional configuration settings:
  - **Primary metric**: Select **Normalized root mean squared error** **
  - **Explain best model**: Selected - *this option causes automated machine learning to calculate **feature importance** for the best model; making it possible to determine the influence of each feature on the predicted label.*
  - **Blocked algorithms**: *Block **all** other than **RandomForest** and **LightGBM** - normally you'd want to try as many as possible, but doing so can take a long time!*
  - Exit criterion:
    - **Training job time (hours)**: 0.5 - *this causes the experiment to end after a maximum of 30 minutes.*
    - **Metric score threshold**: 0.08 - *this causes the experiment to end if a model achieves a normalized root mean squared error metric score of 0.08 or less.*

- Featurization settings:
  - **Enable featurization**: Selected - *this causes Azure Machine Learning to automatically preprocess the features before training.*

3. When you finish submitting the automated ML run details, it will start automatically. Wait for the run status to change from *Preparing* to *Running*.
4. When the run status changes to *Running*, view the **Models** tab and observe as each possible combination of training algorithm and pre-processing steps is tried and the performance of the resulting model is evaluated.



## Review the best model

After the experiment has finished, The best performing model will be generated. We can find it on the ***Details*** tab of the *Automated ML* page.

* Select **Algorithm name**

  See the details of the best model

* ***Metrics***tag 

  shows all the metrics used to evaluate the model and the graphic of them between Predicted and Actual labels.

* ***Explanations*** tab

  * Explanation ID 

    to explore the explanation of the best model

  * **View previouse dashboard experience** 

    * **Global Importance**

      Shows feature importance, how much each feature in the dataset influences the label prediction.



## Deploy the model as service

When get the best model, it can be deployed as a service to the client application to use.



### Deploy a predictive service

There are two method to deploy the model as a service.

* **ACI**: Azure Container Instances

  Recommend to **exercise**, which is suitable deployment target for testing.

* **AKS**: Azure Kubernetes Services

  Recommend to **Production**, which asked to create an *inference cluster compute target.*

==**Steps**==

1. On the **Automated ML** page, select **Display name** of running experiment.

2. On **Details** tag, select **Algorithm name** below **Best model summary**
3. On the top menu of this page, use the **Deploy** button to deploy the model for an example rental bike dataset with the following settings:
   * **Names**: predict-rentals
   * **Decription**: Predict cycle rentals
   * **Compute type**: Azure Container Instance
   * **Enable authentication**: Selected

4. After deployed, in the **Model Summary** section, the **Deploy status** for the **predict-rental** service should be **Running**. (may need to select **Refresh** periodically)

   **Successed**:

   ![Automated Azure ML](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Automated Azure ML.PNG)

5. On the **Endpoints** page and select the **predict-rentals** real-time endpoint. Then select the **Consume** tab, there are information used to connect to your deployed service from a client application.
   - The REST endpoint
   - the Primary Key



### Test the deployed service

1. Open a new browser tab, login the https://ml.azure.com/, view the **Notebooks** page(under **Author**)

2. In the **Notebooks** page, under **My files**, use the **🗋** button to create a new file with the following settings:

   * **File location**: Users/*your user name**
   * **File name**: Test-Bikes.ipynb
   * **File type**: Notebook
   * **Overwrite if already exists**: Selected

3. In the new **Test-Bikes.ipynb**, write the code to test the deploy.***Before runing the notebook, make sure the compute instance is running.***

   ```python
   endpoint = 'http://0ccd8f33-a8d5-4140-a92e-9eb11933411a.eastus2.azurecontainer.io/score'
   key = 'wIXz1irHEoq9xLOsxklTh6CVyde6Z9fA'
   
   import json
   import requests
   
   #An array of features based on five-day weather forecast
   x = [[1,1,2022,1,0,6,0,2,0.344167,0.363625,0.805833,0.160446],
       [2,1,2022,1,0,0,0,2,0.363478,0.353739,0.696087,0.248539],
       [3,1,2022,1,0,1,1,1,0.196364,0.189405,0.437273,0.248309],
       [4,1,2022,1,0,2,1,1,0.2,0.212122,0.590435,0.160296],
       [5,1,2022,1,0,3,1,1,0.226957,0.22927,0.436957,0.1869]]
   
   # Convert the array to JSON format
   input_json = json.dumps({'data': x})
   
   # Set the content type and authentication for the requests
   headers = {"Content-Type":"application/json",
           "Authorization":"Bearer " + key}
   
   # Send the request
   response = requests.post(endpoint, input_json, headers=headers)
   
   # If we got a valid response, display the predictions
   if response.status_code == 200:
       y = json.loads(response.json())
       print('Predictions:')
       for i in range(len(x)):
           print (" Day: {}. Predicted rentals: {}".format(i+1, max(0, round(y["result"][i]))))
   else:
       print(response)
   ```

   ==**Note**==

   * Switch the endpoint to REST endpoint under **Consume** tag.

   * Switch the endpoint to Primary Key under **Consume** tag.



# Clean-up

* Delete **Endpoint**
  * On the **Endpoint** page, select **Endpoint** -> Delete
* Stop **Compute instance** or **Compute Cluster**
  * On the **Compute** page, select **Compute Instance** -> Stop

> **Note**
>
> If you have finished exploring Azure Machine Learning, you can delete the Azure ML workspace and associated resource.
>
> To delete workspacke:
>
> * https://portal.azure.com/, in the **Resource groups** page, open the resource group which to delete
> * Click **Delete resource group**.

