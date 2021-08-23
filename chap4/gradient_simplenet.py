import sys,os
sys.path.append(os.pardir)
import numpy as np
from chap3.functions import softmax
from gradient_method import numerical_gradient
from cross_entropy_error import cross_entropy_error

class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss
    
net=simpleNet()
x=np.array([0.6,0.9])
p=net.predict(x)
t=np.array([0,0,1])