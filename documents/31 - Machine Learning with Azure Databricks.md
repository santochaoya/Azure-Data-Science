# Exploreatory Analysis

## Get data

```python
bostonDF = (spark.read
            .option('HEADER', True)
            .option('inferSchema', True)
            .csv('/mnt/training/bostonhousing.csv'))
					  .drop('_c0')

display(bostonDF)
```



## Explore data

### Count, Mean, and Standard Deviation

```python
display(bostonDF.describe())
```



### Plotting, Distributions, and Outliers

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
		bostonDF
except NameError:
		bostonDF = spark.table('boston')
		
fig, ax = plt.sbuplots()
pandasDF = bostonDF.select('rm', 'crim', 'medv').toPandas()

pd.plottingscatter_matrix(pandasDF)

display(fig.figure)
```



### Correlations

```python
from pyspark.sql.functions import col, rand

bostonWithDummyDataDF = (bostonDF,
                         .select('medv')
                         .withColumn('medvX3', col('medv') * 3)
                         .withColumn('medvNeg', col('medv') * -3)
                         .withColumn('random1', rand(seed=41))
                         .withColumn('random2', rand(seed=44))
                         .withColumn('medvWithNoise', col('medv')*col)
                         .withColumn('medvWithNegativeNoise', col('medv')*col('random1')*-1)
                        )

display(bostonWithDummyDataDF)
```



### Other Visualization Tools

* **Heat maps**
* **Box plots**  - visualizes quantiles and outliers
* **Q-Q Plots** - visualizes two probability distributions
* **Maps and GIS** - visualizes geographically-bound data
* **t-SNE** - plot high demensional data
* **Time series** - plot time-bound variables



Assemble all the ```bostonDF``` into single column ```features```.

```python
from pyspark.ml.feagure import VectorAssembler

assembler = VectorAssembler(inputCols=bostonDF.columns, outputCol='features')
bostonFeatureizedDF = assembler.transform(bostonDF)
```

Calculation correlations across the entire dataset

```python
from pyspark.ml.stat import Correlation

pearsonCorr = Correlation.corr(bostonFeatureizedDF, 'features').collect()[0][0]
pandasDF = pd.DataFrame(peassoCorr.toArray())

pandasDF.index, pandasDF.columns = bostonDF.columns, bostonDF.columns
```

plot heat map

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.heatmap(pandasDF)
display(fig.figure)
```



# Train a model, Predict

## Get data

```python
bostonDF = (spark.read
            .option('HEADER', True)
            .option('inferSchema', True)
            .csv('/mnt/training/bostonhousing.csv'))

display(bostonDF)
```



## Transformation

```python
from pyspark.ml.feature import VectorAssembler

featureCols = ['rm', 'crim', 'lstat']
assembler = VectorAssembler(inputCols=featureCols, outputCol='features')

bostonFeatureizedDF = assembler.transform(bostonDF)

display(bostonFeaturizedDF)
```



## Train a model

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol='medv', featuresCol='features')
# Fit the data to the model
lrModel = lr.fit(bostonFeaturedDF)

print('Coefficients: {0:.1f}, {1:.1f}, {2:.1f}'.format(*lrModel.coefficients))
print('Intercept: {0:.1f}'.format(*lrModel.intercept))
```



## Predictions

### Get testing data

```python
subsetDF = (bostonFeatureziedDF
            .limit(10)
            .select('features', 'medv'))

display(subsetDF)
```



### Predict

```python
predictionDF = lrModel.transform(subsetDF)

display(predictionDF)
```



## Evaluate

```python
from pyspark.ml.linalg import Vectors

data = [(Vectors.denst([6., 3.6, 12.]), )]
predictDF = spark.createDataFrame(data, ['features'])

display(lrModel.transform(predictDF))
```



# Machine Learning workflow

## Split Train and Test Dataset

```python
trainDF, testDF = bostonDF.randomSplit([0.8, 0.2], seed=42)

print('We have {} training examples and {} test examples.'.fromat(trainDF.count(), testDF.count()))
```



## Baseline Model

*Baseline model* offers an educated best guess to improve upon as different models are trained and evaluated. It based on:

* Regression: the average of outcome
* Classification: the model of the data or the most common class

```python
from pyspark.sql.functions import avg, lit

# create a baselie model by calculating the average
trainAvg = trainDF.select(avg('medv')).first()[0]
print('Average home value: {}'.format(trainAvg))

# Take the average calculated on the training dataset and append it as a predicion column
testPredictionDF = test.DF.withColumn('prediction', lit(trainAvg))

display(testPredictionDF)
```



## Evaluation and improvement

Define the evaluator with the prediction column, label column, and MSE metric.

```python
from pyspark.sql.functions import RegressionEvaluator

evaluator = RegerssionEvaluator(predictionCol='prediction', labelCol='medv', metricName='mse')
```

Evaluate ```testPredictionDF``` using the ```.evaluator()```

```python
testError = evaluator.evaluate(testPredictionDF)

print('Error on the test set for the baseline model: {}'.format(testError))
```

