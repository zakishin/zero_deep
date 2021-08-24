import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.random.rand(1000,100)

node_num=100
hidden_layer_size=5
activations={}

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]

    # w=np.random.randn(node_num,node_num)*1
    # w=np.random.randn(node_num,node_num)*0.01

    # Xavierの初期値、前層のノード数をnとして1/√nの標準偏差の分布で初期化
    # Reluの場合はsqrt(2/n)の標準偏差の分布で初期化、Heの初期値
    w=np.random.randn(node_num,node_num)/np.sqrt(node_num)

    z=np.dot(x,w)
    a=sigmoid(z)
    activations[i]=a

for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(),30,range=(0,1))
plt.savefig("histogram.png")