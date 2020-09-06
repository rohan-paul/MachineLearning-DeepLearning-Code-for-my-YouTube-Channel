'''Write a function to find out the prime factors of a number. Example: prime factors of 84 -
[2, 2, 3, 7]'''

import math


def get_prime_factors(number):
    prime_factors = []
    prime_num = 2

    while prime_num <= math.sqrt(number):
        while number % prime_num == 0:
            prime_factors.append(int(prime_num))
            number /= prime_num

        prime_num += 1

    if number > 1:
        prime_factors.append(int(number))

    return prime_factors


print(get_prime_factors(84))
