#!/usr/bin/python
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Bora Banjanin, U of Washington, 2014

import numpy as np
import pylab as plt
import matplotlib as mpl

class DoublePlot():
    """
    Used to plot simultaneous lls

    Usage:
        Pass data and plot upon init
    """

    """
    Global Constants
    """
    r = 1.01

    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None
        self.fx = None
        self.fy = None
        self.fig = None
        self.Lcom = None
        self.Lft = None
        self.Ecom = None
        self.t = None
        self.index = 0

    def hasNext(self):
        if self.index >= 0 \
        and self.index < len(self.x)-1:
            return True
        else:
            return False

    def animIterable(self,i=None):

        if self.x == None \
        or self.y == None \
        or self.theta == None \
        or self.fx == None \
        or self.fy == None \
        or self.fig == None \
        or self.Lcom == None \
        or self.Lft == None \
        or self.Ecom == None \
        or self.t == None:
            raise Exception("DoublePlot: Tried plotting without complete \
            state information")

        if i == None:
            i = self.index

        if i < 0 \
        or i >= len(self.x):
            raise Exception("DoublePlot: Invalid index state")
        #print self.index

        def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
            theta = 2*np.pi/(N-1)*np.arange(N)
            xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
            ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
            return xs, ys

        self.Lcom.set_xdata(self.x[i])
        self.Lcom.set_ydata(self.y[i])
        self.Lft.set_xdata([self.x[i],self.fx[i]])
        self.Lft.set_ydata([self.y[i],self.fy[i]])
        Ex,Ey = Ellipse((self.x[i],self.y[i]), (0.5*self.r, self.r), t=self.theta[i])
        self.Ecom.set_xdata(Ex)
        self.Ecom.set_ydata(Ey)
        self.fig.canvas.draw()
        self.index = i + 1



if __name__ == "__main__":
    #dp = DoublePlot()
    #dp.x = 1
    #dp.test()
    #print dir(dp)
    print "happy"
