import pandas as pd
import numpy as np
import os
import time

import matplotlib.pyplot as plt

from datetime import datetime


def get_dataframe_memory_usage(df):
    """
    Returns the total memory consumption of a Pandas DataFrame in megabytes (MB).
    """
    memory_usage = df.memory_usage(deep=True)
    """ The deep=True parameter is used to include the memory usage of any objects that are referenced by the DataFrame's columns, such as strings or other nested data structures. If you set deep=False, the method will return only the memory usage of the column itself, which may not be the total memory consumed by the DataFrame. """

    # sum the memory usage to get the total memory consumed
    total_memory = memory_usage.sum()

    # convert to megabytes
    total_memory_mb = total_memory / 1048576

    print(f"Total memory consumed by the DataFrame: {total_memory_mb:.2f} MB")


######################################################################

def check_for_missing_timestamp(df):
    # calculate the time differences between consecutive rows
    time_diffs = df.index[1:] - df.index[:-1]

    # count the frequency of each unique time difference
    counts_df = time_diffs.value_counts()

    # return the counts
    return counts_df.head()

"""
btc_df.index[1:] and btc_df.index[:-1] slices the index into two arrays, one starting from the second index (i.e., index 1) and the other ending at the second-to-last index, respectively.

Subtracting these two arrays results in a new array of time differences (in the form of pandas Timedelta objects) between consecutive rows of the dataframe.

The .value_counts() method counts the frequency of each unique time difference and returns a pandas Series object with the counts as values and the unique time differences as indices

## Explanation of `time_diffs = df.index[1:] - df.index[:-1]`

Let's take an example to understand this better. Suppose our pandas dataframe btc_df looks like this:

```py
                       Open    High     Low   Close
Date
2022-01-01 00:00:00  71000  72000   68000   71000
2022-01-02 00:00:00  72000  75000   71000   74000
2022-01-03 00:00:00  75000  76000   73000   75000
2022-01-04 00:00:00  75000  77000   74000   76000


```
### btc_df.index would be:

```
DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'], dtype='datetime64[ns]', name='Date', freq=None)

```

btc_df.index[1:] would be:


```
DatetimeIndex(['2022-01-02', '2022-01-03', '2022-01-04'], dtype='datetime64[ns]', name='Date', freq=None)

```

### btc_df.index[:-1] would be:


```
DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', name='Date', freq=None)

```

### When we subtract these two arrays (btc_df.index[1:] - btc_df.index[:-1]), we get the time difference between each consecutive pair of dates:


```
TimedeltaIndex(['1 days', '1 days', '1 days'], dtype='timedelta64[ns]', name='Date', freq=None)

```

This result is a TimedeltaIndex, which is a pandas data structure that represents a sequence of time durations. In this example, the time duration between each pair of consecutive dates is exactly 1 day, as the dates are daily. """


#################################################################

""" Function to fill in the missing data with the value from the most recent available minute

From above, we can see that we have 78 instances where two consecutive entries are 120 seconds apart, instead of 60 sec, 12 instances which are apart by 180 seconds and so on.

Now, given, this small gaps in data, the simple imputation strategy that we can apply here is that - fill in the missing data with the value from the most recent available minute.

This is what the *method = 'pad'* parameter of the *reindex* function below does.

Instead of the below method, I could have used

btc_df = btc_df.reindex(range(btc_df.index[0],btc_df.index[-1]+60,60), method='pad')

but is more modular and readble


"""

def impute_missing_data(df):
    # sort the index in increasing order
    df = df.sort_index()

    imputed_df = df.reindex(range(df.index[0], df.index[-1]+60, 60),  method='pad')
    """ range(btc_df.index[0],btc_df.index[-1]+60,60) creates a new range of values starting from the first timestamp in the index of btc_df and ending at the last timestamp plus 60 seconds, with a step of 60 seconds. This creates a new index with a minute-by-minute frequency. """

    """ The missing values (i.e., the timestamps that are not present in the original index) are filled in using the method specified by the method parameter. In this case, the pad method is used, which fills the missing values with the most recent available value. """

    # return the imputed dataframe
    return imputed_df


def impute_missing_data(btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing data in the given DataFrame with the value from the most recent available minute.

    Parameters
    ----------
    btc_df : pd.DataFrame
        The input DataFrame containing missing data.

    Returns
    -------
    pd.DataFrame
        The imputed DataFrame with missing values filled in.
    """
    new_index = range(btc_df.index[0], btc_df.index[-1] + 60, 60)
    """ range(btc_df.index[0],btc_df.index[-1]+60,60) creates a new range of values starting from the first timestamp in the index of btc_df and ending at the last timestamp plus 60 seconds, with a step of 60 seconds. This creates a new index with a minute-by-minute frequency. """

    imputed_btc_df = btc_df.reindex(new_index, method='pad')
    """ The missing values (i.e., the timestamps that are not present in the original index) are filled in with the pad method, which fills the missing values with the most recent available value. """

    return imputed_btc_df



"""  the index method is used to retrieve the index (row labels) of a pandas DataFrame or Series object.

## In above method why the 60 is added with btc_df.index[-1]


The reason why 60 is added to btc_df.index[-1] is because the reindex method is being used to create a new index for the dataframe with a minute-by-minute frequency.

In the range function, the start parameter is set to btc_df.index[0], which is the first timestamp in the original index. The end parameter is set to btc_df.index[-1]+60, which is the last timestamp in the original index plus 60 seconds. This creates a new index that starts from the first timestamp in the original index and ends at the last timestamp plus 60 seconds, with a step of 1 minute (i.e., 60 seconds).

For example, suppose we have the following dataframe:

                     value
2022-01-01 00:00:00      1
2022-01-01 00:02:00      2
2022-01-01 00:05:00      3

The index of this dataframe has a frequency of 2 minutes. Now, if we apply the impute_missing_data function to this dataframe, the new index created by the range function would be:

DatetimeIndex(['2022-01-01 00:00:00', '2022-01-01 00:01:00', '2022-01-01 00:02:00', '2022-01-01 00:03:00', '2022-01-01 00:04:00', '2022-01-01 00:05:00', '2022-01-01 00:06:00'], dtype='datetime64[ns]', freq='T')

As you can see, the new index has a minute-by-minute frequency, with timestamps starting from the first timestamp in the original index (i.e., '2022-01-01 00:00:00') and ending at the last timestamp plus 60 seconds (i.e., '2022-01-01 00:05:00' + 60 seconds = '2022-01-01 00:06:00').

"""

########################################################################

# Function to reduce memory usage by converting the data types of each column in the DataFrame to more memory-efficient alternatives.

"""  get_optimal_numeric_type() that takes the minimum and maximum values of the data and the current data type ('int' or 'float') as input, and returns the optimal numeric data type for the given range of values. This helper function simplifies the main reduce_memory_usage() function by encapsulating the logic for determining the optimal data type, making the code more modular and easier to read. """

def get_optimal_numeric_type(c_min: float, c_max: float, col_type: str) -> str:
    """
    Determines the optimal numeric data type for a given range of values.

    Parameters
    ----------
    c_min : float
        The minimum value of the data.
    c_max : float
        The maximum value of the data.
    col_type : str
        The current data type of the column ('int' or 'float').

    Returns
    -------
    optimal_type : str
        The optimal data type for the given range of values.
    """
    type_info = np.iinfo if col_type == 'int' else np.finfo
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        if col_type in str(dtype):
            if c_min > type_info(dtype).min and c_max < type_info(dtype).max:
                return dtype
    return None

""" Based on the data type and the range of values, the function determines the smallest possible data type that can accommodate the data without losing information. For example, if the data type is an integer and the range of values fits within the bounds of an int8 data type, the function converts the column data type to int8: """

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces memory usage of a pandas DataFrame by converting its columns to the most memory-efficient data types
    without losing information.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame that needs memory optimization.

    Returns
    -------
    df : pd.DataFrame
        The optimized pandas DataFrame with reduced memory usage.
    """

    # Iterate through each column in the DataFrame
    df_copy = df.copy()
    for col in df_copy.columns:
        col_type = df_copy[col].dtype

        # Check if the data type is not an object (i.e., numeric type)
        if col_type != object:
            c_min, c_max = df_copy[col].min(), df_copy[col].max()
            col_type_str = 'int' if 'int' in str(col_type) else 'float'
            optimal_type = get_optimal_numeric_type(c_min, c_max, col_type_str)
            if optimal_type:
                df_copy[col] = df_copy[col].astype(optimal_type)
        # If the data type is an object, convert the column to a 'category' data type
        else:
            df_copy[col] = df_copy[col].astype('category')

    # Return the optimized DataFrame with reduced memory usage
    return df_copy
""" ## Whats does the np.iinfo and np.finfo do

np.iinfo and np.finfo are two utility functions provided by NumPy to obtain machine limits for integer and floating-point types, respectively.

**`np.iinfo:`** This function returns the machine limits for integer types. It takes an integer data type as input (e.g., np.int8, np.int16, np.int32, or np.int64) and returns an object containing the minimum and maximum values that the specified data type can represent. You can access these values using the min and max attributes of the returned object. For example:

```py
int_info = np.iinfo(np.int32)
print(int_info.min)  # -2147483648
print(int_info.max)  # 2147483647

```

**`np.finfo:`** This function returns the machine limits for floating-point types. It takes a floating-point data type as input (e.g., np.float16, np.float32, or np.float64) and returns an object containing information about the specified data type, such as the minimum and maximum representable positive values, the smallest representable positive number greater than zero (machine epsilon), and the number of decimal digits of precision. You can access these values using the attributes of the returned object. For example:

```py
float_info = np.finfo(np.float32)
print(float_info.min)    # -3.4028235e+38
print(float_info.max)    # 3.4028235e+38
print(float_info.eps)    # 1.1920929e-07
print(float_info.nmant)  # 23 (number of mantissa bits)
print(float_info.nexp)   # 8 (number of exponent bits) """


#################################################

def create_lagged_dataframe(df: pd.DataFrame, lag_values: list = [3, 2, 1]) -> pd.DataFrame:
    """
    Create a lagged DataFrame for time series data preparation.

    Args:
        df (pd.DataFrame): The input time series DataFrame.
        lag_values (list): The list of lag values for shifting the DataFrame. Default is [3, 2, 1].

    Returns:
        pd.DataFrame: The concatenated lagged DataFrame.
    """
    shifted_dfs = [df.shift(lag) for lag in lag_values]  # List comprehension to create a list of shifted DataFrames
    shifted_dfs.append(df)  # Append the original DataFrame to the list

    lag_df = pd.concat(shifted_dfs, axis=1)  # Concatenate the list of shifted DataFrames and the original DataFrame along the columns axis

    return lag_df