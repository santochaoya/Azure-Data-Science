This document notes that the capabilities of Azure Databricks and the Apache Spark notebook for processing huge files.

* Azure Databricks platform
* Create Azure Databricks workspace
* Create a notebook insider folder in Databricks
* Fundamentals of Apache Spark notebook
* Create or attach to Spark cluster
* types of tasks well-suited to Apache Spark



# Azure Databricks

*Azure Databricks* is a fully managed, cloud-based Big Data and Machine Learning platform.



## What is Apache Spark?

Spark is a unified processing engine that can analyze big data using SQL, machine learning, graph processing. or real-time stream analysis



# Create an Azure Databricks workspace and cluster

## Databricks

### Deploy an Azure Databricks workspace

1. Open the Azure portal.
2. Click **Create a Resource** in the top left
3. Search for "Databricks"
4. Select *Azure Databricks*
5. On the Azure Databricks page select *Create*
6. Provide the required values to create your Azure Databricks workspace:
   - **Subscription**: Choose the Azure subscription in which to deploy the workspace.
   - **Resource Group**: Use **Create new** and provide a name for the new resource group.
   - **Location**: Select a location near you for deployment. For the list of regions that are supported by Azure Databricks, see [Azure services available by region](https://azure.microsoft.com/regions/services/).
   - **Workspace Name**: Provide a unique name for your workspace.
   - **Pricing Tier**: **Trial (Premium - 14 days Free DBUs)**. You must select this option when creating your workspace or you will be charged. The workspace will suspend automatically after 14 days. When the trial is over you can convert the workspace to **Premium** but then you will be charged for your usage.
7. Select **Review + Create**.
8. Select **Create**.

The workspace creation takes a few minutes. During workspace creation, the **Submitting deployment for Azure Databricks** tile appears on the right side of the portal. 



## Cluster

1. When your Azure Databricks workspace creation is complete, select the link to go to the resource.
2. Select **Launch Workspace** to open your Databricks workspace in a new tab.
3. In the left-hand menu of your Databricks workspace, select **Clusters**.
4. Select **Create Cluster** to add a new cluster.

1. Enter a name for your cluster. Use your name or initials to easily differentiate your cluster from your coworkers.
2. Select the **Cluster Mode**: **Single Node**.
3. Select the **Databricks RuntimeVersion**: **Runtime: 7.3 LTS (Scala 2.12, Spark 3.0.1)**.
4. Under **Autopilot Options**, leave the box **checked** and in the text box enter `45`.
5. Select the **Node Type**: **Standard_DS3_v2**.
6. Select **Create Cluster**.

![databricks1](/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/Images/databricks1.png)



# 　Azure Databricks Notebooks

## Create

1. In the Azure portal, click **All resources** menu on the left side navigation and select the Databricks workspace you created in the last unit.
2. Select **Launch Workspace** to open your Databricks workspace in a new tab.
3. On the left-hand menu of your Databricks workspace, select **Home**.
4. Right-click on your home folder.
5. Select **Create**.
6. Select **Notebook**.
7. Name your notebook **First Notebook**.
8. Set the **Language** to **Python**.
9. Select the cluster to which to attach this notebook.

## Attach detach notebook 

### Attach

From the right-top selection

![databricks2](/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/Images/databricks2.png)



### Detach

* Detach your notebook from the cluster

- Restart the cluster
- Attach to another cluster
- Open the Spark UI
- View the log files of the driver

