from math import sqrt


def mean(numbers):
    if len(numbers) > 0:
        return sum(numbers) / len(numbers)
    return float('NaN')

