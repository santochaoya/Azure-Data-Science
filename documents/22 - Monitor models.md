This document is about to use **Azure Application Insights** to monitor a deployed Azure ML model.

The Learning objectives are:

* Enable Application Insights monitoring for an Azure ML web service
* Capture and view model telemetry



# Enable Application Insights

To log telemetry in application insights, we must have an Azure Application Insights resource associated with our Azure ML workspace, and configure the service to use it for telemitry logging.



## Associate Application Insights with Workspace

When we create a workspace, the associated application insights will be created in the same resource. We can view it on the **Overview** page, or use SDK.

```python
from azureml.core import Workspace

ws = Workspace.from_config()
ws.get_details()['applicationInsights']
```



## Configure Application Insights for a service

```python
dep_config = AciWebservce.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               enable_app_insight=True)
```

When a service is already deployed, if we prefer to enable the Application Insights, we can modify the deployment configuration for AKS, or update by SDK.

```python
service = ws.webservices['my-svc']
service.update(enable_app_insights=True)
```



# Capture and view telemetry

Application Insights will automatically captures any information written to the standard output and error logs, provides a query capability to view data.

## Write log data

```python
def init():
		global model
		model = joblib.load(Model.get_model_path('my_model'))
		
def run(raw_data):
		data = json.loads(raw_data)['data']
		predictions = model.predict(Data)
    log_txt = 'Data: ' + str(data) + ' - Predictions' + str(predictions)
    print(log_txt)
    
   	return predictions.tolist()
```



## Query logs in Application Insights

Using the Log Analytics query interface for Application Insights in the Azure Portal with a SQL-like query syntax.

