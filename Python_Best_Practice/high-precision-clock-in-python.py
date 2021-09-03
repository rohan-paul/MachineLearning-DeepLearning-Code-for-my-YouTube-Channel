"""
When need to print a high-Precision timestamp.

If I tried to get the current time of my system with microsecond precision I really dont get to the precision level that I want

"""
import time
from datetime import datetime as dt

x = dt.now()
t = 2 ** 10000
y = dt.now()
print(x, "\n", y)

"""
Will give you  something like below:

2021-09-03 02:28:28.763578
2021-09-03 02:28:28.763602

Note they are almost same.

So to resolve the above lack of precision, we can depend onf time.time_ns() -

which was introduced in Python 3.7 as new functions to the time module providing higher resolution:

https://docs.python.org/3/library/time.html - Similar to time() but returns time as an integer number of nanoseconds since the epoch.

"""


print("\n Convert to High Precision nanoseconds timestamp \n")

time1 = time.time_ns()
print(time1)

time2 = time.time_ns()
print(time2)

# print(time1, "\n", time2)

print("\n Convert to Human Readable TimeStamp \n")


# How many nanosecond in 1 second? The answer is 1000000000.
# Convert to floating-point seconds
t1 = time1 // (10 ** 9)
t2 = time2 // (10 ** 9)


dt1 = dt.fromtimestamp(t1)
dt2 = dt.fromtimestamp(t2)

# Finally Human Readable form
human_readable_time1 = dt1.strftime("%Y-%m-%d %H:%M:%S")
human_readable_time2 = dt2.strftime("%Y-%m-%d %H:%M:%S")

print(human_readable_time1)
print(human_readable_time2)
