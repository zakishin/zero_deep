import numpy as np
import matplotlib.pyplot as plt
import step_function
import relu
def sigmoid(x):
    return 1/(1+np.exp(-x))

# x=np.arange(-5.0,5.0,0.1)
# y1=sigmoid(x)
# y2=step_function.step_function(x)
# y3=relu.relu(x)
# plt.plot(x,y1)
# plt.plot(x,y2,linestyle='--')
# plt.plot(x,y3,linestyle='dashdot')
# plt.ylim(-0.1,1.1)
# plt.show()