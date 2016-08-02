#MathUtils

a collection of math-related functions

    npm install mathutils

###Currently available functions

#####isEven(n) / isOdd(n)
Check if the bit for `2^0` is set. If it is, `n must be odd.

#####powermod(a, b, c)
powermod is a way of computing `(a ^ b) mod c` without having to deal with giant numbers that would loose their precision.

#####slowIsPrime(n)
Returns if `n` is a prime. Extremely slow, but absolutely accurate.

#####fastIsPrime(n)
Retuns if a `n` is a prime. Based upon [Fermat's little theorem](http://en.wikipedia.org/wiki/Fermat%27s_little_theorem).

Note: Doesn't take care of carmichael primes, so you probably want to use this in combination with slowIsPrime.

#####isPrime(n)
Runs both `fastIsPrime` and `slowIsPrime`. This way, it manages to be both (relatively) fast and accurate.

#####randomPrime(length)
Returns a pseudo-random prime number (based on `Math.random`). `length` defaults to 3.

#####gcd(a, b)
Returns the greatest common divisor of `a` and `b`. Based on [Euclids algorithm](http://en.wikipedia.org/wiki/Euclid%27s_algorithm)

#####egcd(a, b)
Computes the [extended Euclidean algorithm](http://en.wikipedia.org/wiki/Extended_Euclidean_algorithm). Returns an array `[d, s, t]`.

    gcd(a, b) = d = s * a + t * b

#####modularInverse(a, b)
Returns the [modular multiplicative inverse](http://en.wikipedia.org/wiki/Modular_multiplicative_inverse) of `a` and `b`.

###TODO
* write tests
* add more documentation
* learn how to breakdance