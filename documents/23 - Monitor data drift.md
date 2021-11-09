Monitoring the data drift to provent the reducing accuracy from changing trends in data over time. Over time may be trends to change the profile of data. But it might make the model less accurate.

*Data Drift* is known as the change in data profiles between training and inferencing. 

**Learning objectives**:

* Create a data drift monitor
* Schedule monitoring
* View monitoring results



# Create a data drift monitor

## Monitor data drift by comparing datasets

It's common for organizations to continue to collect new data after a model has been trained. Then compare the growing collection of new data to the original training data, and identify any changing data trands that might affect model accuracy.

There are two dataset need to be registered:

* *baseline dataset* - the original training data
* *target dataset* - will be compared to the baseline based on time intervals.This dataset requires a column for each feature you want to compare, and a timestamp column so the rate of data drift can be measured.



### Create a monitor

```python
from azureml.datadrift imoprt DataDriftDetector

monitor = DataDriftDetector.create_from_datasests(workspace=ws,
                                                  name='dataset-drift-detector',
                                                  baseline_data_set=train_ds,
                                                  target_data_set=new_data_es,
                                                  compute_target='aml-cluster',
                                                  frequency='Week',
                                                  feature_list=['age', 'height', 'bmi'],
                                                  latency=24)
```



### Backfill to compare

```python
import datatime as dt

backfill = monitor.backfill(df.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())
```



# Scheduling alerts

After creating monitors can schedule alerts and Additionally, you can specify a threshold for the rate of data drift and an operator email address for notifications if this threshold is exceeded.

## Configure data drift monitor schedules

Data drift monitoring works by running a comparison at scheduled **frequency**, and calculating data drift metrics for the features in the dataset that you want to monitor. You can define a schedule to run every **Day**, **Week**, or **Month**.

For dataset monitors, you can specify a **latency**, indicating the number of hours to allow for new data to be collected and added to the target dataset. For deployed model data drift monitors, you can specify a **schedule_start** time value to indicate when the data drift run should start (if omitted, the run will start at the current time).

* ```frequency``` - a comparison running frequency

* ```latency``` - how many hours for new data
* ```schedule_start``` data drift run start time



## Configure alerts

We can send notification automatically when a specific value of data drift detected.

```python
alert_email = AlertConfiguration('data_scientists@contoso.com')
monitor = DataDriftDetector.create_from_datasets(ws, 'dataset-drift-detector', 
                                                 baseline_data_set, target_data_set,
                                                 compute_target=cpu_cluster,
                                                 frequency='Week', latency=2,
                                                 drift_threshold=.3,
                                                 alert_configuration=alert_email)
```

