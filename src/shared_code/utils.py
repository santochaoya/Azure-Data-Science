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
