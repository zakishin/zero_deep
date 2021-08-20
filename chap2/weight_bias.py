import numpy as np
x=np.array([0,1])
w=np.array([0.5,0.5])
b=-0.7
tmp=np.sum(w*x)+b
if tmp>=0:
    print(1)
else:
    print(0)
