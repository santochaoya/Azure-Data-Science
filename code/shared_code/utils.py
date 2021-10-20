import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype


def add_statistics_line(figure, feature, metrics):
    """Add a statistic line on graphic with a specific required method.

    :param figure: subplot figure the line need to be added
    :param feature: Pandas Series
        Examine feature
    :param metric: List
        a list contains required statistic metrics
    :return: graphic
        A line of required statistic metric
    """

    # if not is_numeric_dtype(feature):
    #     raise TypeError(
    #         "Please use a numeric value as the inputting feature."
    #     )

    for metric in metrics:
        if 'min' in metric:
            figure.axvline(x=feature.min(), color='gray', linestyle='dashed', linewidth=2)
        if 'mean' in metric:
            figure.axvline(x=feature.mean(), color='red', linestyle='dashed', linewidth=2)
        if 'median' in metric:
            figure.axvline(x=feature.median(), color='blue', linestyle='dashed', linewidth=2)
        if 'mode' in metric:
            figure.axvline(x=feature.mode()[0], color='yellow', linestyle='dashed', linewidth=2)
        if 'max' in metric:
            figure.axvline(x=feature.max(), color='gray', linestyle='dashed', linewidth=2)

def show_distribution(feature):
    """Show histogram and boxplot to display the distribution and statistics of a specific feature.

    :param feature: pandas series
        Examine feature
    :return: distribution graphic of feature
    """

    if not is_numeric_dtype(feature):
        raise TypeError(
            "Please use a numeric value as the inputting feature."
        )

    # Get statistics
    min_val = feature.min()
    max_val = feature.max()
    mean_val = feature.mean()
    median_val = feature.median()
    mode_val = feature.mode()[0]

    print(feature.name, '\nMinimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                                             mean_val,
                                                                                                             median_val,
                                                                                                             mode_val,
                                                                                                             max_val))

    # Create a figure for 2 subplots(2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the histogram
    ax[0].hist(feature, color='green')

    # Add statistics
    add_statistics_line(ax[0], feature, ['min', 'mean', 'median', 'mode', 'max'])

    # Customize histogram
    ax[0].set_ylabel('Frequency')

    # Plot the boxplot
    ax[1].boxplot(feature, vert=False)

    # Customize boxplot
    ax[1].set_xlabel('Value')

    # Add title
    plt.suptitle(feature.name)

    # Show the figure
    plt.show()

def show_density(feature):
    """Using line graphic to show the probability density function .

    :param features: Pandas series
        Examine feature, works on single feature.
    :return: graphic
        Probability density function graphic with some statistics metrics.
    """

    if not is_numeric_dtype(feature):
        raise TypeError(
            "Please use a numeric value as the inputting feature."
        )

    # plot the density graphic
    fig = plt.figure(figsize=(10, 4))

    # density function
    feature.plot.density(color='green')

    # Add statistics
    add_statistics_line(plt, feature, ['mean', 'median', 'mode'])

    # Show the graphic
    plt.show()

def trim_outliers_maximum(df, feature, max_percentile):
    """Trim outliers based on percentiles

    :param feature: pandas series
        columns need to trim outliers
    :param max_percentile: float
        represent percentage to trim the maximal outliers
    :return: pandas series
        columns trim outliers
    """

    return df[df[feature] < df[feature].quantile(max_percentile)]

def trim_outliers_minimum(df, feature, min_percentile):
    """Trim outliers based on percentiles

    :param feature: pandas series
        columns need to trim outliers
    :param min_percentile: float
        represent percentage to trim minimal outliers
    :return: pandas series
        columns trim outliers
    """

    return df[df[feature] > df[feature].quantile(min_percentile)]

