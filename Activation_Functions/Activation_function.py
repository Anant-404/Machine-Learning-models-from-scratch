import numpy as np

def step_function(x):
    # if(x>=0):
    #     return 1
    # else:
    #     return 0
    return np.where(x>=0,1,0);#where(condition , if true , if false)