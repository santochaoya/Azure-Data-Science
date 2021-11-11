This is the document about Machine Learning flow to track experiment, log metrics and compare runs.



* **Required Library**

```mlflow==1.7.0```



# MLflow Tracking

*MLflow Tracking* si a logging API specific for. machine learning and agnostic to libraries and environments that do the training.

## Track Runs

Each run can record the following information:

* **Parameters: ** key-value pairs of input parameters, such as the number of tress in a random forest model
* **Metrics**: evaluation metrics such as RMSE, MSE, R2 or ROC curve.
* **Artifacts**: arbitrary output files in any format, such as images, data files, and pickled models.
* **Source:** The code to ran the experiment



> **NOTE**: MLflow can only log PipelineModels.

```python
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

mlflow.set_experiment(f'/Users/{username}/tr-mlflow')

with mlflow.start_run(run_name="LR-Single-Feature") as run:
  	# Define pipeline
    vecAssembler = VectorAssembler(inputCols=['bedrooms'], outputCol='features')
    lr = LinearRegression(featuresCol='features', labelCol='price')
    pipeline = Pipeline(stages=[vecAssembler, lr])
    pipelineModel = pieline.fit(trainDF)
    
    # log parameters
    mlflow.log_param('label', 'price-bedrooms')
    
    # log model
    mlflow.spark.log_model(pipelineModel, 'model')
    
    # Evaluate predictions
    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluad(predictionCol='prediction', labelCol='price', metricName='rmse')
    rmse = regressionEvaluator.evaluate(predDF)
    
    # Log metrics
    mlflow.log_metrics('rmse', rmse)
    
# display_run_uri(run.info.experiment_id, run.info.run_id)
```



Build a linear regression model use all of the features

```python
from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name='LR-ALL-Features') as run:
  	# Create a pipeline
    rFormula = 	RFormula(formula='price ~ .', featuresCol='features'， labelCol='price', handleInvalid='skip')
    lr = LinearRegression(labelCol='price', featuresCol='features')
    pipeline = Pipeline(stages=[rFormula, lr])
    pipelineModel = pipeline.fit(trainDF)
    
    # Log pipeline
    mlflow.spark.log_model(pipelineModel, 'model')
    
    # Log parameter
    mlflow.log_param('label', 'price-all-features')
    
    # Create predictions and metrics
    predDF = pipelineModel.transform(testDF)
    regressionEvaluater = RegressionEvaluator(labelCol='price', predictionCol='prediction')
    rmse = regressionEvaluator.setMetricName('rmse').evaluate(predDF)
    
    # Log both metrics
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    
# display_run_uri(run.info.experiment_id, run.info.run_id)
```



Logging artifacts to keep a visual of log normal histogram. Due to the log normal distribution of price, will try to predict the log of the price.

```python
from pyspark.ml.feature import RFormula
from pyspark.sql.functions import col, log, exp
import matplotlib.pyplot as plt

with mlflow.start_run(run_name='LR-Log-Price') as run:
  	# Take log of price
    LogTrainDF = trainDF.withColumn('log_price', log(col('price')))
    logTestDF = testDF.withColumn('log_price', log(col('price')))
    
    # Log parameters
    mlflow.log_param('label', 'log_price')
    
    # Create pipeline
    rFormula = RFormula(formula='log_price ~ . - price',
                        featuresCol='features',
                        labelCol='log_price',
                        handleInvalid='skip')
    lr = LinearRegression(labelCol='log_price',
                          predictionCol='log_prediction')
    pipeline = Pipeline(stages=[rFormula, lr])
    pipelineModel = pipeline.fit(logTrainDF)
    
    # Log model
    mlflow.spark.log_model(pipelineModel, 'log-model')
    
    # Make predictions
    predDF = pipelineModel.transform(logTestDF)
    expDF = predDF.withColumns('prediction', exp(col('log_prediction')))
    
    # Evaluate predictions
    rmse = regressionEvaluator.setMetricName('rmse').evaluate(expDF)
    r2 = regressionEvaluator.setMetricName('r2').evaluate(expDF)
    
    # Log metrics
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    
    # Log artifact
    plt.clf()
    logTrainDF.toPandas().hist(column='log_price', bins=100)
    fig_path = username + 'logNormal.png'
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    display(plt.show())
```



# Querying Past Runs

Query the past run programmaticalluy in order to use this data back in Python.Using ```MlflowClient```

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.list_experiments()
```



Using ```search_runs``` to find all runs for a given experiment

```python
experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)
```



Pull the last run and look at metrics

```python
runs = client.search_runs(experiment_id, order_by=['attributes.start_time desc'],
                          max_results=1)
run_id = runs[0].info.run_id

runs[0].data.metrics
```



# Load Saved Model

```python
loaded_model = mlflow.spark.load_model(f'runs/{run.info.run_uuid}/log-model')
display(loaded_model.transform(testDF))
```



# Log Param, Metrics, and Artifactsblu

```python
def generate_plot():
		import numpy as np
		import matplotlib.pyplot as plt
		
		np.random.seed(1939487)
		
		fig, ax = plt.subplots()
		for color in ['tab:blue', 'tab:orange', 'tab:green']:
      	n = 750
        x, y = np.random.rand(2, n)
        ax.scatterIx, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')
        
    ax.legend()
    ax.grid(True)
    
    return fig, plt
  
generate_plot()
```



