# Clustering

Clustering is a process of grouping objects with similar object based on their data value or features. The difference between clustering and classification is that clustering is an unsupervised model. It identifys examples from a similar collection of features. The goal of clustering is to find an optimal way to split the dataset into groups, in which find a way to identify groups of entities that are close to the similar one and separated from other groups..



> We have an example of a seed dataset with six dimensions.
>
> ![clustering1](/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/documents/images/clustering1.png)
>
> 



## PCA - *Principal Component Analysis*

To translate the multiple dimensions feature value into a two dimension dataset. By analysising the relationship between features, summarize each observation as a coordinates for two principle components.

```python
# Identify features
features = data[data.columns[0:6]]

from sklearn.preprocssing import MinMaxScaler
from sklearn.decomposition import PCA

# Normalize the numeric features
scaled_features = MinMaxScaler().fit_transform(features)

# Get two principle components
pca = PCA(N_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
feature_2d[0:10]

# Visualize the principle components
import matplotlib.pyplot as plt

plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()
```



## Clusters

### Identify number of clusters

If we have obvious clues to see distinct groups, we can try to start with a series of clustering models with an incremental number of clusters. and measure how tightly the data poins are grouped with each cluster. The metric used to measure is WCSS(*Within cluster sum of squares*).

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create 10 models from 1 to 10 clusters
wcss = []
for i in range(1, 11):
		# KMeans model
		kmeans = KMeans(n_clusters = i)
		# Fit th model
		kmeans.fit(features)
		# Get the WCSS value
		wcss.append(kmeans.inertia_)
		
# Plot the wcss
fig = plt.figure(figsize=(10, 4)
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('WCSS by Clusters')
plt.show()

```

The Griphic of WCSS with different number of clusters as below:

![clustering2](/Users/xiao/Projects/git/Microsoft-Azure-Data-Science/documents/images/clustering2.png)

From the graphic we can see, the optimal number of clusters we choose might be 3. The plot has a large reduction in WCSS from 1 to 2, and a further noticable reduction from 2 to 3. After that, the reduction is not pronounced. *This is a good indecation that there are 2 to 3 reasonably well separated clusters of data points.*



## Different types of clustering models

### K-Means

* **K** : the number of clusters
* **Means**: Using *Mean Distance* to identify the centroids in each iteration.

**Steps of K-Means**

1. The feature values are vectorized to define the n-dimensions coordinate(n is the number of features)
2. Using WCSS to choose the number of clusters(K). Randomly choose K points as centroid of each cluster. 
3. Assign each point to the nearest centroid.
4. Move each centroid to the center of each cluster based on the mean distances between the points.
5. Iterally step 4 and 5 until the clusters become stabel or reach the maximal number of iteration.

```python
from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)

# Fit the model and predict the clusters
km_clusters = model.fit_predict(features)

# Plot the clusters
import matplotlib.pyplot as plt

def plot_clusters(samples, clusters):
		col_dict = {0:'blue', 1:'green', 2:'yellow'}
		mrk_dict = {0:'*', 1:'x', 2:'+'}
		colors = [col_dict[x] for x in cluters]
		markers = [mrk_dict[x] for x in clusters]
		for sample in range(len(clusters)):
				plt.scatter(samples[sample][0], samples[sample][1], color=colors, marker=markers)
        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Kmeans Clusters')
    plt.show()
    
plot_clusters(features_2d, km_clusters)
```



### Hierarchical Clustering

Hierarchical Clustering dosen't require to identify a number of clusters in advance, and can sometimes provide a more interpretable results. But the major drawback of hierarchical clustering is that these approaches can take much longer to compute and sometimes are not stables for large datasets.



**Steps of Hierarchical Clustersing**

1. Compute the linkage distance between each data point

2. Combine the points with nearest neighbor as clusters

3. Computer the linkage distance between each cluster

4. Combine the clusters with nearest neighbor

5. Repeat step 3 and 4 until all data points are in a single cluster

   

**Linkage Mehods**

* **Ward linkage**: 

  The increase of variance between two clusters. Only chose Euclidian Distance

* **Average Linkage**

  Mean pairwise distance between the members of the two clusters

* **Complete/Maximal Linkage**

  The maximum distance between the members of the two clusters

  

**Distance Methods**

* **Euclidian or L2 Distance**
* **Manhattan or L1 Distance**
* **Cosine Sililarity**



**Method of Hirarchical Clustering**

* **Divisive** - **TOP DOWN**

* **Agglomerative** - **BOTTOM UP**

  

```python
from sklearn.cluster import AgglomerativeClustering

# Fit the Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)

# Plot the clustering
plot_clusters(features_2d, agg_clusters)
```



### Applications

* Using clustering to separate customers into distinct segments
* Using clustering to label the dataset, then apply classification model to predict more samples



