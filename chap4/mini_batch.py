import sys,os
sys.path.append(os.pardir)
import numpy as np
import mnist

(x_train, t_train), (x_test, t_test) =mnist.load_mnist(normalize=True,one_hot_label=True)

train_size=x_train.size[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]

def cross_entropy_loss(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size