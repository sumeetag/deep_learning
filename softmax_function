#Softmax converts the scores into respective probability. 

"""Softmax."""
from math import exp
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #print (type(x))
    if type(x) is list:
        
        sum_e = 0
        arr = []
        for i in x:
            sum_e += exp(i)
            
        for i in x:
            arr.append(exp(i)/sum_e)
       
        return arr
    else:
        
        arr = []
        for i in x.T:
            sum_e = 0
            arr1 = []
            for j in i:
                sum_e += exp(j)
            
            for j in i:
                arr1.append(exp(j)/sum_e)
            
            arr.append(arr1)
            
        aa = np.asarray(arr)
        
        return aa.T
                
    pass  # TODO: Compute and return softmax(x)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
#print type(scores)
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
