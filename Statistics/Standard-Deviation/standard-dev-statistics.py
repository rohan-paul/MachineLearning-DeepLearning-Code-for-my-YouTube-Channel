# Youtube Link => https://youtu.be/w2WrskWX60o

import statistics

arr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

print(statistics.stdev(arr))

# 1.3693063937629153


''' statistics.stdev uses Sampling Standard Deviation. i.e the denominator of the equation is N - 1 instead of N '''
