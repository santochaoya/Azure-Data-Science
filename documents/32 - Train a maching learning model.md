This document is about to train a machine learning model with Azure Databricks.

Learning objectives:

* Transformers, estimators, and pipelines of PySpark
* Build pipelines for common data featurization tasks
* Regression modeling



# Featurization

## Transformers, Estimators, and Pipelines

* **Transformers**

  * ```transform()```

    > return a new DataFrame with one or more columns appended to it

* **Estimator**

  * ```.fit()```

    > return a model

* **Pipeline**

  * ```.fit()```

    > combine transformer and estimators together and make it easier to combine multiple algorithms.



There are some featurization approaches:

* Encoding categorical variables
* Normalizing
* Create new features
* Handling missing values
* Binning/discretizing



## Encoding

### Get data

```python
airbnbDF = spark.read.parquet('/mnt/training/airbnb/sf-listings/sf-listings-correct-type.parquet')

display(airbnbDF)
```



### Unique value and indexing

```python
from pyspark.ml.feature import StringIndexer

uniqueTypesDF = airbnbDF.select('room_type').distinct()

indexer = StringIndexer(inputCol='room_type', outputCol='room_type_index')
indexerModel = indexer.fit(uniqueTypesDF)
indexedDF = indexerModel.transform(uniqueTypesDF)

display(indexedDF)
```



### One-Hot Encoding

```python
from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols=['room_type_index'], outputCols=['encoded_room_type'])
encoderModel = encoder.fit(indexedDF)
encodedDF = dncoderModel.transform(indexedDF)

display(encodedDF)
```



## Missing or Null value 

* **Dropping**
* **Adding a placeholder** - e.g. -1
* **Basic imputing** - offen use mean of non-missing value
* **Advanced imputing** - advanced stragegies such as clustering, oversampling



### Dropping missing values

```python
countWithoutDropping = airbnbDF.count()
countWithDropping = airbnbDF.na.drop(subset=['zipcode', 'host_is_superhost']).count()

print('Count without droppping missing values: {}'.format(countWithoutDropping))
print('Count with droppping missing values: {}'.format(countWithDropping))
```



### Imputing with median

```python
from pyspark.sql.functions import Imputer

imputeCols = ['host_total_listings_count',
              'bathrooms',
              'beds',
              'review_scores_rating',
              'review_scores_accuracy',
              'review_scores_cleanliness',
              'review_scores_checkin',
              'review_scores_communitcation',
              'review_scores_location',
              'review_scores_value']

imputer = Imputer(strategy='median', inputCols=imputeCols, outputCols=imputeCols)
imputerModel = imputer.fit(airbnbDF)
imputedDF = imputerModel.transform(airbnbDF)

display(imputedDF)
```



## Creating a pipeline

Spark uses the convention established by ```scikit-learn``` to combine each of these steps into a single pipeline

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer,
                            encoder,
                            imputer])
```

The pipeline itself is a estimator

```python
pipelineModel = pipeline.fit(airbnbDF)
transformedDF = pipelineModel.transform(airbnbDF)

display(transformedDF)
```



# Regression Model

## Get dataset

```python
bostonDF = (spark.read
            .option('HEADER', True),
            .option('inferSchema', True)
            .csv('/mnt/training/bostonhousing/bostonhousing/bostonhousing.csv')
            .drop('_c0'))

display(bostonDF)
```

Create a column ```features``` that has a single input variable ```rm``` by using ```VectorAssmbler```

```python
from pyspark.ml.feature import VectorAssembler

featureCol = ['rm']
assembler = VectorAssembler(imputCols=featureCol, outputCol='features')

bostonFeaturizedDF = assembler.transform(bostonDF)

display(bostonFeaturizedDF)
```



## Fit a linear regression model

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol='features', labelCol='medv')

lrModel = lr.fit(bostonFeaturizedDF)
```



## Model Interpretation

* What did the model estimate my codfficients to be?
* Are coeffcients statistically significant?
* How accurate was my model?

```python
print('The intercept is: {}\nThe coefficient for rm are {}'.format(lrModel.intercept, *lrModel.coefficients))
```



### Significant coefficients

* **p-value**

```python
summary = lrModel.summary
summary.pValues
```

* **$$R^2$$**

```python
summary.r2
```

* Take a look at the ```summary``` attribute of ```lrModel```

```python
[attr for attr in dir(summary) if attr[0] != "_"]
```



## Multivariate Regression

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Featurization
featureCols = ['rm', 'crim', 'lstat']
assemblerMultivariate = VectorAssembler(inputCols=featureCols, outputCol='features')
bostonFeaturizedMultivariateDF = assemblerMultivariate.transform(bostonDF)

# Train the model
lrMultivariate = (LinearRegression()
                  .setLabelCol('medv')
                  .setFeaturesCol('features'))

lrModelMultivariate = lrMultivariate.fit(bostonFeaturizedMultivariateDF)
summaryMultivariate = lrModelMultivariate.summary

# Evaluate
print('The intercept is {}'.format(lrModelMultivariate.intercept))
for i, (col, coef) in enumerate(zip(featureCols, lrModelMultivariate.coefficients)):
  	print('Coefficient for {} is {}'.format(col, coef))
    
print('R2 score: {}'.format(summaryMultivariate.r2))
```



