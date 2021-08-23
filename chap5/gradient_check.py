from http.client import NETWORK_AUTHENTICATION_REQUIRED
import sys,os
sys.path.append(os.pardir)
import numpy as np
from two_layer_net import TwoLayerNet
from datasets.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =load_mnist(normalize=True,one_hot_label=True)
print(t_train.shape)
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

x_batch=x_train[:3]
t_batch=t_train[:3]

grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradient(x_batch,t_batch)

for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+':'+str(diff))
