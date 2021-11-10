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



### Step 2: Optimize layout for fast queries:

```python
OPTIMIZE events
WHERE data >= current_timestamp() - INTERVAL 1 day
ZORDER BY (eventType)
```



## Basic syntax

### Upsert Syntax

* INSERT row
* if the row existing, UPDATE the row

```sql
MERGE INTO customers --Delta table
USING updates
ON customers.customerId = source.customerId
WHEN MATCHED THEN
		UPDATE SET address = updates.address
WHEN NOT MATCHED
		THEN INSET (customerId, address) VALUES (updates.customerId, updates.address)
```



### Time Travel Syntax

Because Delta Lake is version controlled, we have the option to query past versions of the data.

Example of using time travel to reproduce experiments and reports:

```sql
SELECT count(*) FROM events
TIMESTAMP AS OF timestamp

SELECT count(*) FROM events
VERSION AS OF version
```

```python
spark.read.format('delta').option('timestampAsOf', timestamp_string).load('/events/')
```

If need to rollback

```sql
INSERT INTO my_table
		SELECT * FROM my_table TIMESAMPE AS OF
		data_sub(current_date(), 1)
```



## Create

Using ```dataframe.write.format('delta').save('/data')``` to create a delta lake.

```python
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

inputSchema = StructType([
  	StructField('InvoiceNo', IntegerType(), True),
  	StructField('StockCode', IntegerType(), True)
])

rawDataDF = (spark.read
             .option('header', 'ture'),
             .schema(inputSchema)
             .csv(inputPath))

# write to Delta Lake
rawDataDF.write.mode('overwrite').format('delta').partitionBy('Country').save(DataPath)
```

```python
display(spark.sql("SELECT * FROM delta.'{}' LIMIT 5".format(DataPath)))
```



## Create a table using Delta Lake

```python
spark.sql("""
		DROP TABLE IF EXISTS customer_data_delta
""")

spark.sql("""
		CREATE TABLE customer_data_delta
		USING DELTA
		LOCATION '{}'
""".format(DataPath))
```



## Metadata

When we have data back in ```customer_data_delta``` in place, this table in the **Hive metastore** automatically inherits the schema, partitioning, and table properties of the existing data.

The actual schema is stored in the ```_delta_log```

```python
display(dbutils.fs.ls(DataPaht + '/_delta_log'))
```



### Display metadata

```sql
DESCRIBE DETAIL customer_data_delta
```



## APPEND Using Delta Lake

```python
(newDataDF.write.format('delta').partitionBy('Country').mode('append').save(DataPath))
```



# Managed Delta Lake

