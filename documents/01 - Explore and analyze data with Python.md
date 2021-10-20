# NumPy and Pandas



## NumPy array VS. List

The NumPy array support mathematical operations on numeric data. The type of NumPy array shows as ndarray. The 'nd' represents that this is a structure that can consist of multiple dimensions.



## Explore data

> There are an example of the grades and study hours per week from the students for a data science class.  The initial DataFrame as below:
>
> ```python
> import numpy as np
> import pandas as pd
> 
> # grades of students
> data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
> grades = np.array(data)
> 
> # Define an array of study hours
> study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
>                13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
> 
> # Create a 2d array of grades and study hours
> student_data = np.array([study_hours, grades])
> 
> # Create DataFrame to contains all information
> df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
>                                      'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
>                                      'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
>                             'StudyHours':student_data[0],
>                             'Grade':student_data[1]})
> ```



### Finding and filtering data in DataFrame

* **loc**

  Find specific rows or columns by index value.

  ```python
  df_students.loc[1, ['Grade', 'Name']]
  ```

  or

  ```python
  df_students.loc[df_students['Name'] == 'Aisha']
  ```

  equal to

  ```python
  df_students[df_students['Name'] == 'Aisha']
  ```

  equal to

  ```python
  df_students.query('Name' == 'Aisha')
  ```

  equal to

  ```python
  df_students[df_students.Name = 'Aisha']
  ```

* **iloc**

  Find rows and columns by position.

  ```python
  df_students.iloc[0:5]
  ```



### Loading a DataFrame from a file

* **CSV**

```python
pd.read_csv('grade.csv', delimeter=',', header='infer')
```



### Handing missing value

* **Filter missing value**

  * Identify which individual values are null in DataFrame

    ```python
    df_students.isnull()
    ```

  * Total amount of missing values for each columns

    ```python
    df_students.isnull().sum()
    ```

  * Filter the rows contains null in any of the columns

    ```python
    df_students[df_students.isnull().any(axis=1)]
    ```

* **Fill missing values**

  * Fill missing value with mean of the columns

    ```python
    df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
    ```

  * remove rows or columns which contains missing value

    ```python
    df_students = df_students.dropna(axis=0, how='any')
    ```



### Explore data

* **mean**

  ```python
  mean_study = df_students.StydyHours.mean()
  ```

  or
  
  ```python
  mean_study = df_students['StudyHours'].mean()
  ```
  
* **add pandas series as a new columns**

  ```python
  passes = pd.Series(df_students['Grade'] > 60)
  df_students = pd.concat([df_students, passes.rename('Pass')], axis=1)
  ```

* **Aggregate to count**

  * amount of pass and not pass

    ```python
    df_students.groupby(df_students.pass).Name.count()
    ```

  * aggregative amount of grades and study hours

    ```python
    df_students.groupby(df_students.pass)['StudyHours', 'Grade'].mean()
    ```

* **Sorting**

  ```python
  df_students.sort_values('Grade', ascending=False)
  ```



## Visualize Data

### Representing data visually

Graphing data can be useful for understanding data, finding outlier data, understanding how numbers are distributed, and so on.

> Visualize data from previous example, getting packages as below:
>
> ```python
> from matplotlib import pyplot as plt
> ```
>
> 

* **Create figure and subplots**

  * Figure

    ```python
    fig = plt.figure(figsize=(10, 5))
    ```

  * Subplots(1 row, 2 columns)

    ```python
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ```

    

* **Bar Chart**

  ```python
  plt.bar(x=df_students.Name, height=df_students.Grade, color='green')
  ```

  

* **Pie Chart**

  Typically, pie chart used to represent the percentage or amount of classifications.

  ```python
  pass_counts = df_students['Pass'].value_counts()
  plt.pie(pass_counts, labels=pass_counts)
  ```

  

* **Customize chart**

  ```python
  plt.title('Student Grades')
  plt.xlabel('Students')
  plt.ylabel('Grades')
  plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=1)
  plt.xticks(rotation=90)
  ```



* Examples

  ```python
  # create a figure for 2 subplots
  fig, ax = plt.subplots(1, 2, figsize=(12, 4))
  
  # create a bar chart of name vs grade on the first axis
  ax[0].bar(x=df_students.Name, height=df_students.Grade, color='green')
  ax[0].set_title('Grades')
  ax[0].set_xticklabels(df_students.Name, rotation=90)
  
  # create a pie chart of pass counts on the second axis
  pass_counts = df_students['Pass'].value_counts()
  ax[1].pie(pass_counts, labels=pass_counts)
  ax[1].set_title('Passing Grades')
  ax[1].legend(pass_counts.keys().tolist())
  
  # Add a title to the Figure
  fig.suptitle('Student Data')
  
  # Show the figure
  plt.show()
  ```



***What's more?***

DataFrame provides its own plotting method.

```python
df_students.plot.bar(x='Name', y='Grade', color='green', figsize=(12, 4))
```



### Statistical Analysis

Visualize data to explore the distribution and statistics of features.

* **Histogram**

  The histogram is more interested in the shape of the distribution, the variances and potential outliers of features. Each bar represents the frequency of numerical variables.
  
  ```python
  # plot histogram figure
  var_data = df_students['Grade']
  plt.hist(var_data)
  
  # add line of statistics
  plt.axvline(x=var_data.min(), color='gray', linestyle='dashed', linewidth=2)
  plt.axvline(x=var_data.max(), color='gray', linestyle='dashed', linewidth=2)
  plt.axvline(x=var_data.mean(), color='red', linestyle='dashed', linewidth=2)
  ```
  
  



* **Boxplot**

  Focus on the quartiers, the range, and outliers rather than the shape of distribution.

  ```
  # plot boxplot
  var_data = df_students['Grade']
  plt.boxplot(var_data)
  ```
  



* Example

  ```python
  # The feature to examine
  var_data = df_students['Grade']
  
  # Create a function that we can re-use
  def show_distribution(var_data):
      '''
      This function will make a distribution (graph) and display it
      '''
  
      # Get statistics
      min_val = var_data.min()
      max_val = var_data.max()
      mean_val = var_data.mean()
      med_val = var_data.median()
      mod_val = var_data.mode()[0]
  
      print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val, mean_val, med_val, mod_val, max_val))
  
      # Create a figure for 2 subplots (2 rows, 1 column)
      fig, ax = plt.subplots(2, 1, figsize = (10,4))
  
      # Plot the histogram   
      ax[0].hist(var_data)
      ax[0].set_ylabel('Frequency')
  
      # Add lines for the mean, median, and mode
      ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
      ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
      ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
      ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
      ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)
  
      # Plot the boxplot   
      ax[1].boxplot(var_data, vert=False)
      ax[1].set_xlabel('Value')
  
      # Add a title to the Figure
      fig.suptitle('Data Distribution')
  
      # Show the figure
      fig.show()
  
  show_distribution(df_students['Grade'])
  ```
  
  
  
* **Probability Density Function**

  ```python
  fig = plt.figure(figsize=(12, 4))
  
  # plot density
  var_data.plot.density()
  
  # Stiatistics
  plt.axvline(x=var_data.mean(), color='gray', linestyle='dashed', linewidth=2)
  plt.axvline(x=var_data.median(), color='gray', linestyle='dashed', linewidth=2)
  plt.axvline(x=var_data.mode(), color='red', linestyle='dashed', linewidth=2)
  
  # show the figure
  plt.show()
  ```

  ### TODO: MORE READING REGARDING TO PROBABILITY DENSITY FUNCTION

  

# Examine Real World Data

## Real world tips:

- Check for missing values and badly recorded data

- Consider removal of obvious outliers

- Consider what real-world factors might affect your analysis and consider if your dataset size is large enough to handle this

- Check for biased raw data and consider your options to fix this, if found

  

## Summary

Here we've looked at:

1. What an outlier is and how to remove them
2. How data can be skewed
3. How to look at the spread of data
4. Basic ways to compare variables, such as grades and study time



> This chapter will talk about study time data from previous instance.

## Distribution

### Original data

![EDA1](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA1.PNG)

From distribution above, we can see the whiskers of the box plot begins at around 6.0, indicating that the vast majority of the first quarter of the data is above this value. The minimal **o**, indicating it is statistically an outlier.

For learning purposes here, we treated the value **o** as an outlier and exclued it. 

> In the real world problem, we unusual exclude the outlier when your sample size is too small. This is because the smaller our sample size is, it is more likely that our sampling is a bad representation of the whole population.

Here we can use a technical to exclude the observations below a percentage.

```python
q01 = df_students.StudyHours.quantile(0.01)
study_hours = df_students[df_students.StudyHours > q01]['StudyHours']
```

![EDA2](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA2.PNG)

From the data distribution we can see, the distribution is not symmetric, there are some students have extremely high value of study hours of 16 hours. but the bulk of data is between 8 and 13 hours.

### Density

![EDA3](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA3.PNG)

The kind of density distribution is right skewed. The mass of the data is on the left side of the distribution. Some extremely high value makes a long tail to the right; which pull the mean to the right.

### Comparing variables

* **numeric and categorical**

  Comparing the study hours to the pass category.

   ![EDA4](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA4.PNG)

  It is apparent that students who passed the course tended to study for more hours than students who didn't pass the cours.

  

* **numeric and numeric**

  * **Bar Chart**

    Comparing the study hours and grades after scaling

    ![EDA5](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA5.PNG)

    After normalized, it's easier to see an apparent relationship between grade and study hours. It is not an exact match, but it's definitely seems like that students with higher grades tend to have studied more.

    

  * **Scatter Plot**

    Comparing the correlation between two numeric variables is to use scatter plot.

  ![EDA6](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\EDA6.PNG)

  It looks like there is a discernible pattern in which the students who studied the most hours are also the students who got the highest grades.
