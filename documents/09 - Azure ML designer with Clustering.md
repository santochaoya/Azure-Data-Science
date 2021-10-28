This document is about how to use Azure Machine Learning designer to train and publish a clustering model.

Most of steps will be the same as the Regression model. This document only determine the difference steps.



# Train the Model

After splitting the dataset into training and testing dataset, we use a module named ==**Train Clustering Model**== with all features selected to the model.

> Note: The clustering model should assign clusters to the data items by using all of the features you selected from the original dataset.

1. Drag a ==**Train Clustering Model**== module and connect the *Result dataset1* (left) output of the ==**Split Data**==
2. On the **Column set** tab, select **all columns**
3. Drag a ==**K-Means Clustering**== module and connect its output to the **Untrained model** (left) input of the ==**Train Clustering Model**== module.
   * Set the **Number of centroids** parameter to **3**.
4. Drag an ==**Assign Data to Clusters**== module and connect to the output of ==**Train Clustering Model**== and ==**Split Data**==

