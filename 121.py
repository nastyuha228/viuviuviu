from tetet import biba
import random
import math


I = [[1.0], [0.1], [0.9]]
targets_list = [[0.5], [0.8], [0.5]]
random.seed(42)

net = biba(3, 3, 3, 0.1)
print("до тренировки:")
print(net.query(I))
net.train(I, targets_list, epochs=10000)


print("После тренировки:")
print(net.query(I))

