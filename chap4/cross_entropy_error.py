import numpy as np
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# t=[0,0,1,0,0,0,0,0,0,0]
# y=[0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
# print(cross_entropy_error(np.array(y),np.array(t)))