import numpy as np


# initial an 6*6 matrix
na = np.arange(1, 49).reshape(3, 4, 4)

print(na) # show na

print("Show the shape of na: ",na.shape)

print("======")
print(na[0,1,3])
print("======")
print(na[:,:2,:2])
print("======")
print(na[::2,::2,::2])
print("======")
print(na[-3::2,-3::2,-3::2])




