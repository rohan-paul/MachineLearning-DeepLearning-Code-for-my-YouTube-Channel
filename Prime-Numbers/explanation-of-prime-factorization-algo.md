The Algorithm / Steps

1> I start the divisor to be the smallest prime number, which is 2. Divide by 2 as many times as I can, until I can no longer divide by 2. (And during this process, record the number of times I can successfully divide.)

A few more words on Why I am dividing or adjusting the initial number in the above e.g. number /= prime_num

Say, if I do not divide - prime factors with greater than 1 multiplicity will result in the larger factor being non-prime. I dont want to consider about 2*n or 3*n because I have already checked and captured in my prime_factor list for 2 and 3

So, to solve the above problem, I keep dividing the larger factor of a pair by the smaller one until it no longer divides.

2> So with each successive while loop I am dividing the number by successively larger primes until I find one that is a factor of the number.

That is after the first division by 2 (assuming I found 2 to be a factor of the initial number), take the result from (1) i.e. the adjusted number after dividion by 2, and divide by 3 as many times as I can.

Then go to the next prime number, 5. Take the result from (2), and divide by 5 as many times as I can.

And during this successive divisions, as soon as I have found a factor or divisor, p , I can replace n with m=n/p and continue the process of trial division with primes greater than or equal to p up to (n/p)^1/2

Repeat the process, until final result is 1.

---

Now the most important part for improving time-complexity of the algorithm - which follows from the principle that -

In prime factorization of n the loop goes upto square root of n and not till n.

3> However, we don't need to go out that far i.e. upto the number itself. If we've tested all the primes up to the square root of our target number without finding a divisor, we don't need to go any further because we know that our target number is prime after all.

This is because - If a number N has a prime factor larger than √n , then it surely has a prime factor smaller than √n

So, it's sufficient to search for prime factors in the range [1, sqrt(n)], and then use them in order to compute the prime factors in the range [sqrt(n), n].

The way we implement the above is as follows -

A. We know that prime factors with >1 multiplicity will result in the larger factor being non-prime. To solve this, keep dividing the larger factor of a pair by the smaller one until it no longer divides.

---

To explain a little more on this - If you do not find a factor less than sqrt(n), then the number "n" itself is a prime number.

Because, consider the opposite, you find two factors larger than sqrt(n), say "a" and "b".

But then a \* b > will be larger than the original number itself, because both a and b are greater than the sqrt(n) - BUT this is impossible.

Therefore, if there is a factor larger than sqrt(n), there must also exist a factor smaller than sqrt(n), otherwise their product would exceed the value of "n".

But in this case it will make the variable 'prime_num' a NON-PRIME number - Because now it can be expressed as ( a _ b ) which is only possible for NON-Prime number, as a Prime number can only be expressed as 1 _ itself.

But here we are ONLY interested in finding Prime factors, and so will ignore all NON-PRIME FACTORS.

Hence, in our Algorithm, checking all the possible prime-number upto the math.sqrt(number) is sufficient