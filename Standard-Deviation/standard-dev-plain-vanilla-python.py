from math import sqrt


def mean(numbers):
    if len(numbers) > 0:
        return sum(numbers) / len(numbers)
    return float('NaN')


def std_dev(numbers):
    if len(numbers) > 0:
        avg = mean(numbers)
        variance = sum([(i - avg) ** 2 for i in numbers]) / len(numbers)
        standard_deviation = sqrt(variance)
        return standard_deviation
    return float('NaN')


arr = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

print(std_dev(arr))

# 1.2909944487358056
