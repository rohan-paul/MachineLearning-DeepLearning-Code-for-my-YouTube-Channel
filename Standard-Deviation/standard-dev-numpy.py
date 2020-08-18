import numpy as np

arr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

print(np.std(arr))

''' Numpy uses Population Standard Deviation by default. i.e the denominator of the equation in N instead of N - 1 '''

# 1.2909944487358056

print(np.std(arr, ddof=1))

# 1.3693063937629153

'''
The NumPy function np.std takes an optional parameter ddof: "Delta Degrees of Freedom". By default, this is 0. Set it to 1 to get the result for Sampling Standard Deviation:

if we select a random sample of N elements from a larger distribution and calculate the variance, division by N can lead to an underestimate of the actual variance. To fix this, we can lower the number we divide by (the degrees of freedom) to a number less than N (usually N-1). The ddof parameter allows us change the divisor by the amount we specify.

https://numpy.org/doc/stable/reference/generated/numpy.std.html

So a simplistic rule of thumb is:

Include ddof=1 if you're calculating np.std() for a sample taken from your full dataset.

And else

If you are calculating on the full dataset and NOT a sample of it, then use ddof=0 i

The DDOF is included for samples in order to counterbalance bias that can occur in the numbers.
'''
