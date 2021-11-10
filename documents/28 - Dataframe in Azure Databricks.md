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

