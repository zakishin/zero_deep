import numpy as np
class dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None
    
    def forward(self,x,train_fig=True):
        if train_fig:
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask

        else:
            return x*(1.0-self.dropout_ratio)

    def backward(self,dout):
        return dout*self.mask