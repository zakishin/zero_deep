def And(x,y):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x*w1+y*w2
    if tmp<=theta:
        return 0
    else:
        return 1
print(And(1,0))