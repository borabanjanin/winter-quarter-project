import numpy as np

class MyBounds(object):
    def __init__(self, xmax=[1.1,1.1], xmin=[-3,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

mybounds = MyBounds()
kwargs = {'x_new':[-2,0]}
print mybounds(**kwargs)
