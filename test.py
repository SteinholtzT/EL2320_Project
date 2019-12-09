

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

mx = 100
my = 100

s = np.array([  [np.random.randint(mx, size=10)],  # x
                [np.random.randint(my, size=10)],  # y 
                [np.random.randint(10, size=10)]                      ,      # dx
                [np.random.randint(10, size=10)]                      ,])    # dy
s = np.reshape(s, (4, 10))
s = s.astype(int)


x = s[0,:]
print("x", x)
print()
s[0,:] = np.sum([s[0,:], s[2, :]], axis=0)
s[1,:] = np.sum([s[1,:], s[3, :]], axis=0)

test = s[2,:] - x
print(test)
print()
test2 = np.divide(test, 2)
print(test2)
test2 = test2.astype(int)
print(test2)