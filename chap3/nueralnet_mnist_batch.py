from http.client import NETWORK_AUTHENTICATION_REQUIRED
import pickle
import sys,os

from numpy.lib.function_base import bartlett
sys.path.append(os.pardir)
import mnist
import numpy as np
import softmax
import sigmoid
from PIL import Image

def get_data():
    (x_train, t_train), (x_test, t_test) =mnist.load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("C:/Users/shinya/Desktop/code/zero_deep/chap3/sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def predict(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,w1)+b1
    z1=sigmoid.sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid.sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax.softmax(a3)
    return y

x,t=get_data()
network=init_network()

batch_size=100
accuracy_cnt=0

for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
    
print('Accuracy:'+str(accuracy_cnt/len(x)))
