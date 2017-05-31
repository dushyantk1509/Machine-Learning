from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm,variance,step=2,correlation=False):
    var = 1
    ys = []
    for i in range(hm):
        y = var + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            var+=step
        elif correlation and correlation == 'neg':
            var-=step
    xs = [i for i in range(len(ys))]

    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)

#xs,ys = create_dataset(40,40,2,correlation='pos')

def best_fit_para(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs)*mean(ys)) - mean(xs*ys)) )
    b = mean(ys) - m*mean(xs)
    return m,b

m,b = best_fit_para(xs,ys)

regressionline = [(m*x)+b for x in xs]

y_mean_line = [mean(ys) for y in ys]
#print(y_mean_line)
print(xs,ys)

plt.scatter(xs,ys)
#plt.plot(xs, regressionline)
#plt.plot(ys, y_mean_line)
#plt.show()
