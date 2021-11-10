Learning objectives:

* Create UDF
* Articulate performance advantages of Vectorized UDF in Python



# User Defined Functions - UDF

* function: 

```python
def firstInitialFunction(name):
		return name[0]
  
firstInitialFunction('Jane')
```



## Define a UDF

```python
firstInitialUDF = udf(firstInitialFunction)
```

```python
from pyspark.sql.function import col
display(airbnbDF.select(firstInitialUDF(col('host_name'))))
```

or

Create a UDF using ```spark.sql.register```

```python
airbnbDF.createOrReplaceTempView('airbnbDF')
spark.udf.register('sql_udf', firstInitialFunction)
```



## Decorator Syntax

Define a UDF using decorator syntax in Python with the dataType and function will return.

However a local Python function can not be called anymore.

```python
# Our input/output is string
@udf('string')
def decoratorUDF(name):
		return name[0]
  
display(airbnbDF.select(firstInitialUDF(col('host_name'))))
```



## Drawbacks

* UDFs cannot be optimized by Catalyst Optimizer
* The function **has to be serialized** and sent out to the executors
* we have to **sign up a Python interpreter** on every Executor to run the UDF



## Vectorized UDF

Vectorized UDFs utilize Apache Arrow to speed up computations.

```python
from pyspark.sql.functions import pandas_df

@pandas_udf('string')
def vectorizedUDF(name):
		return name.str[0]
```

Register Vectorized UDFs to the SQL namespace

```python
spark.udf.register('sql_vectorized_udf', vectorizedUDF)
```

