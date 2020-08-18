import pandas as pd

arr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

pandas_series = pd.Series(arr)

print(pandas_series.std())

# 1.3693063937629153


''' Pandas uses Sampling Standard Deviation by default. i.e the denominator of the equation is N - 1 instead of N '''