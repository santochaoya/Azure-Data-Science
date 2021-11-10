This document will note the objectives:

* Read multiple file types by Azure Databricks, both with and without a Schema
* Combine imputs from files and data stores, such as Azure SQL Database
* Transform and sotre data for advanced analytics

from [document](https://docs.microsoft.com/en-us/learn/modules/read-write-data-azure-databricks/2-read-data-csv-format)



# Read data in CSV format

## Create resource

### Deploy an Azure Databricks workspace

Create an Azure Databricks workspace on [Deploy Databricks from the ARM Template](portal.azure.com/#create/Microsoft.Template)

1. Provide the required values to create your Azure Databricks workspace:
   - **Subscription**: Choose the Azure Subscription in which to deploy the workspace.
   - **Resource Group**: Leave at Create new and provide a name for the new resource group.
   - **Location**: Select a location near you for deployment. For the list of regions supported by Azure Databricks, see [Azure services available by region](https://azure.microsoft.com/regions/services/).
   - **Workspace Name**: Provide a name for your workspace.
   - **Pricing Tier**: Ensure `premium` is selected.
2. Accept the terms and conditions.
3. Select Purchase.
4. The workspace creation takes a few minutes. 



### Create a cluster

1. When your Azure Databricks workspace creation is complete, select the link to go to the resource.
2. Select **Launch Workspace** to open your Databricks workspace in a new tab.
3. In the left-hand menu of your Databricks workspace, select **Clusters**.
4. Select **Create Cluster** to add a new cluster.
5. Enter a name for your cluster. Use your name or initials to easily differentiate your cluster from your coworkers.
6. Select the **Cluster Mode**: **Single Node**.
7. Select the **Databricks RuntimeVersion**: **Runtime: 7.3 LTS (Scala 2.12, Spark 3.0.1)**.
8. Under **Autopilot Options**, leave the box **checked** and in the text box enter `45`.
9. Select the **Node Type**: **Standard_DS3_v2**.
10. Select **Create Cluster**.



### Clone the archive

1. If you do not currently have your Azure Databricks workspace open: in the Azure portal, navigate to your deployed Azure Databricks workspace and select **Launch Workspace**.
2. In the left pane, select **Workspace** > **Users**, and select your username (the entry with the house icon).
3. In the pane that appears, select the arrow next to your name, and select **Import**.
4. In the **Import Notebooks** dialog box, select the URL and paste in the following URL:

```html
 https://github.com/solliancenet/microsoft-learning-paths-databricks-notebooks/blob/master/data-engineering/DBC/03-Reading-and-writing-data-in-Azure-Databricks.dbc?raw=true
```

1. Select **Import**.
2. Select the **03-Reading-and-writing-data-in-Azure-Databricks** folder that appears.



## Complete the following notebook

Open the **1.Reading Data - CSV** notebook. Make sure you attach your cluster to the notebook before following the instructions and running the cells within.

Within the notebook, you will:

- Start working with the API documentation
- Introduce the class `SparkSession` and other entry points
- Introduce the class `DataFrameReader`
- Read data from:
  - CSV without a Schema
  - CSV with a Schema

After you've completed the notebook, return to this screen, and continue to the next step.



# Read data in JSON format



Open the **2.Reading Data - JSON** notebook. Make sure you attach your cluster to the notebook before following the instructions and running the cells within.



# Read data in Parquet format

Open the **3.Reading Data - Parquet** notebook. Make sure you attach your cluster to the notebook before following the instructions and running the cells within.



# Read data stored in tables and views

Open the **4.Reading Data - Tables and Views** notebook. Make sure you attach your cluster to the notebook before following the instructions and running the cells within.



# Write data

Open the **5.Writing Data** notebook. Make sure you attach your cluster to the notebook before following the instructions and running the cells within.



Now that you have concluded this module, you should know:

1. Read data from CSV files into a Spark Dataframe
2. Provide a Schema when reading Data into a Spark Dataframe
3. Read data from JSON files into a Spark Dataframe
4. Read Data from parquet files into a Spark Dataframe
5. Create Tables and Views
6. Write data from a Spark Dataframe



# Clean up

If you plan on completing other Azure Databricks modules, don't delete your Azure Databricks instance yet. You can use the same environment for the other modules.

### Delete the Azure Databricks instance

1. Navigate to the Azure portal.
2. Navigate to the resource group that contains your Azure Databricks instance.
3. Select **Delete resource group**.
4. Type the name of the resource group in the confirmation text box.
5. Select **Delete**.

