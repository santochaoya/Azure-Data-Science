*Azure Delta Lake* is a transactional storage layer designed specifically to work with Apache Spark and Databricks File System(DBFS). The *core* of Delta Lake is an optimized **Spark table**, that **stores data as Apache Parquet files in DBFS** **and maintains a transaction log**.



# Data Lakes

A *data lake* is a storage repository that inexpensively stores a vast amount of raw data, both current and historical, in native formats such as XML, JSON, CSV and Parquet. These data is not ready for data science & ML. These are unreliable data.



# Delta Lake - Makes data ready for analytics.

We can read and write data in Delta lake through Apache Spark SQL batch and streaming APIs. These are the same familiar APIs that used to work with Hive tables or DBFS directories.



## Using Delta with exisiting Parquet tables

### Step 1: Convert ```Parquet``` to ```Delta``` tables

```python
CONVERT TO DELTA parquet. 'path/to/table' [NOSTATISTICS]
[PARTITIONED BY (col_name1 col_type, col_name2 col_type2, ....)]
```

