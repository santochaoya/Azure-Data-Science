This is about model selection with Spark



# Tuning, Validating and Saving

## Hyperparameter Tuning

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder


bostonDF = (spark.read
           	.option('HEADER', True)
            .option('inferSchema', True)
            .csv('/mnt/training/bostonhousing/bostonhousing/bostonhousing.csv')
            .drop('_c0'))

trainDF, testDF = bostonDF.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=bostonDF.columns[:-1], outputCol='features')

lr = (LinearRegression()
      .setLabelCol('medv')
      .setFeaturesCol('features'))

pipeline = Pipeline(stage=[assembler, lr])
print(lr.explainParams())

# Hyperparameter Tuning
paramGrid = (ParamGridBuilder()
             .addGrid(lr.maxIter, [1, 10, 100])
             .addGrid(lr.fitIntercept, [True, False])
             .addGrid(lr.standardization, [True, False])
             .build())
```



### Evaluate

Create a ```RegressionEvaluator()``` to evaluate the grid search experiments and a ```CrossVlidator()``` to build our model.

```python
from pyspark.ml.evaluation import RegressionEvaluator
from pysaprk.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol='medv',
                                predictionCol='prediction')

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evalutor,
                    numFolds=3,
                    seed=42)

# Fit the cross validation
cvModel = cv.fit(trainDF)

# Look at the socre from each experiment
for params, score in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics):
  	print(''.join([param.name+"\t"+str(params[param])+'\t' for param in params]))
    print('\tScore: {}'.format(score))
    
# Access the best model
bestModel = cvModel.bestModel
```



# Save models and predictions

```python
model_path = userhome + '/cvPipelineModel'
dbutils.fs.rm(model_path, recurse=True)

cvModel.bestModel.save(model_path)
```



* **look at the path**

```python
dbutils.fs.ls(model_path)
```



* **Save predictions made on ```testDF```**

```python
predictions_path = userhome +'/modelPredictions.parquet'
cvModel.bestModel.tranform(testDF).write.mode('OVERWRITE').parquet(predictions)
```

