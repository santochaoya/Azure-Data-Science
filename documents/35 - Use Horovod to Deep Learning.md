This is about to run a long-running, distributed deep learning training jobs on Spark with Horovod.



# Horovod

*HorovodRunner* is a neneral API to run distribuited DL workloads on Databricks using Uber's Horovod framework.

## Build Model

```python
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(42)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():
  	return Sequential([Dense(20, input_dim=8, activation='relu'),
                       Dense(20, activation='relu'),
                       Desen(1, activation='linear')])
```



## Shared Data

