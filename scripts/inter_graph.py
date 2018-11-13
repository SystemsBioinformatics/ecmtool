from time import sleep

import matplotlib.pyplot as plt
from matplotlib import interactive
from random import randint

def fun1(x):
    return 2 * x + 4

def fun2(x):
    return 0.5 * x + 1

interactive(True)

plt.figure(1)
plt.subplot(111)
plt.scatter([x for x in range(1, 4)], [fun1(x) for x in range(1, 4)])

plt.pause(0.0001)

plt.figure(2)
plt.subplot(111)
plt.scatter([x for x in range(1, 4)], [fun2(x) for x in range(1, 4)])
plt.pause(0.0001)

for x in range (4, 20):
    for i in range(1, 3):
        plt.figure(i)
        plt.subplot(111)
        plt.scatter([x], [fun1(x) if i == 1 else fun2(x)])
        plt.pause(0.0001)
        sleep(1)

interactive(False)
plt.show()

