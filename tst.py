import numpy as np

N = 10**5
test = 0
for i in range(N):
    test += np.random.binomial(n=1, p=0.8)

print(test / N)


test2 = 0
for i in range(N):
    test2 += int(np.random.uniform() < 0.8)

print(test2 / N)

test3 = set()
for i in range(N):
    test3.add(np.random.binomial(n=1, p=0.5) * 2 - 1)

print(test3)
