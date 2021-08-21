import numpy as np
def mean_squared_loss(y,t):
    return 0.5*np.sum((y-t)**2)

# t=[0,0,1,0,0,0,0,0,0,0]
# y=[0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
# print(mean_squared_loss(np.array(t),np.array(y)))