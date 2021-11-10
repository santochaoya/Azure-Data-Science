## Learning objectives

In this module, you will learn some build-in functions to handle the dataframe.



# The data souce

We can use ```dbutils.fs.ls()``` to **view** our data on the DBFS.

```python
(source, sasEntity, sasToken) = getAzureDataScource()

spark.conf.set(sasEntity, sasToken)
path = source + '/wikipedia/pagecounts/staging_parquet_en_only_clean/'
files = dbutils.fs.ls(path)
display(files)
```



# Create a DataFrame

Using the ```spark``` with the instance of ```SparkSession``` and entry point to Spark 2.0 applications to read the Parquet files into DataFrame. From there we can access the ```read``` object which gives us an instance of ```DataFrameReader```.

```python
parquetDir = source + '/wikipedia/pagecounts/staging_parquet_en_only_clean/'

pagecountsEnALLDF = (spark	# SparkSession & Entry Point
                     .read	# DataFrameReader
                     .parquet(parquetDir)	# Returns an instance of DataFrame
                     )
```

 

# Build-in Functions 

## Describe the data

```count()```

> Returns the number of rows in Dataset.

```python
total = pagecountsEnAllDF.count()

print('Record Count: {0:, }'.format(total))
```



## Cache

```cache()``` or ```persist()```

> Cache data for archieving better performance with Apache Spark.

This is because every action requres Spark to read the data from its source(Azure Blob, Amazon S3, HDFS, etc) but caching moves that data into the memory of the local executor for 'instant' access.

```cache()``` is just an alias for ```persist()```

```python
(pagecountsEnAllDF
	.cache()	# Mark the DataFrame as cached
	.count()  # Materialize the cache
)
```

After that, when running ```pegecountsEnAllDF.count()```, it should take significantly less time.



### Remove a cache

```unpersist()``` from ```DataFrame```



## Data Type

 ```printSchema()```

> Return the type of each column in DataFrame.



## Check the dataframe

```show()```

> Display the data. on the console, It is an action and will triiger a job

* ```n``` : The number of records to print
* ```truncate``` : If ```True```, columns. wider than 20 characters will be truncated.

```python
pagecountsEnAllDF.show()
```



```display()```

> A more powerful displaying function. It is an action and will trigger a job.



## Transformation

```limit()```

> Return a **new** DataFrame by taking the first n rows.
>
>  It returns a new ```DataFrame``` and will not trigger a job.

```python
limitedDF = pagecountsEnAllDF.limit(5)
```



```select()```

> Return a new dataset by computing the given column expression for each element.
>
>  It returns a new ```DataFrame``` and will not trigger a job.

```python
onlyTreeDF = (pagecountsEnAllDF
              .select('project', 'article', 'requests'))

onlyTreeDF.printSchema()
```



```drop()```

> Returns a new dataset with a column dropped.
>
>  It returns a new ```DataFrame``` and will not trigger a job.

```python
droppedDF = (pagecountsEnAllDF
             .drop('bytes_served'))
doppedDF.printSchema()
```



```distinct()``` or ```dropDuplicates()```

> Returns a new dataset that contains only the unique rows from this dataset.
>
>  It returns a new ```DataFrame``` and will not trigger a job.

```python
distinctDF = (pagecountsEnAllDF
              .select('project')
              .distinct())
```



# DataFrames vs SQL & Temporary Views

* Temporary Views

  ```python
  pagecountsEnAllDF.createOrReplaceTempView('pagecounts')
  ```

* Show SQL

  ```
  tableDF = spark.sql('select distinct project from pagecounts order by project')
  display(tableDF)
  ```

  or

  ```python
  tableDF.show()
  ```

  



