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
        # Here is the crucial part.
        # First quick refreshment on the two key mathematical conjectures of Prime factorization of any non-Prime number
        # Which is - 1. If n is not a prime number AT-LEAST one Prime factor would be less than sqrt(n)
        # And - 2. If n is not a prime number - There can be AT-MOST 1 prime factor of n greater than sqrt(n).
        # Like 7 is a prime-factor for 14 which is greater than sqrt(14)
        # But if the above loop DOES NOT go beyond square root of the initial n.
        # Then how does that greater than sqrt(n) prime-factor
        # will be captured in my prime factorization function.
        # ANS to that is - in my first for-loop I am dividing n with the prime number if that prime is a factor of n.
        # Meaning, after this first for-loop gets executed completely, the adjusted initial n should become
        # either 1 or greater than 1
        # And if n has NOT become 1 after the previous for-loop, that means that
        # The remaining n is that prime factor which is greater that the square root of initial n.
        # And that's why in the next part of my algorithm, I need to check whether n becomes 1 or not,
        prime_factors.append(int(number))

    return prime_factors


print(get_prime_factors(84))


