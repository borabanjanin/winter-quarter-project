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
# (c) Sam Burden, UC Berkeley, 2014

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

    '''
    Class Variables
    '''
    plotNum = 1

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

    @staticmethod
    def plotNumberInc(n=1):
        '''
        .plotNumberInc returns the current figure number and increments
        by default

        INPUTS:
          n - 1 x 1
            The size of the increment is by default 1
        '''
        value = DoublePlot.plotNum
        DoublePlot.plotNum += n
        return value


    def hasNext(self):
        '''
        .hasNext returns whether the iterator has another data point

        OUTPUTS:
            Boolean
        '''
        if self.index >= 0 \
        and self.index < len(self.x)-1:
            return True
        else:
            return False

    @staticmethod
    def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
        theta = 2*np.pi/(N-1)*np.arange(N)
        xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
        ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
        return xs, ys

    def animGenerate(self, o=None, dt=1e-3):
        """
        .animGenerate  generates animation figure and data

        INPUTS:
            o - Obs - trajectory to animate
            dt - time step
        """
        if o is None:
            o = self.obs().resample(dt)

        self.t = np.hstack(o.t)
        self.x = np.vstack(o.x)
        self.y = np.vstack(o.y)
        self.fx = np.vstack(o.fx)
        self.fy = np.vstack(o.fy)
        v = np.vstack(o.v)
        delta = np.vstack(o.delta)
        self.theta = np.vstack(o.theta)
        dtheta = np.vstack(o.dtheta)
        PE = np.vstack(o.PE)
        KE = np.vstack(o.KE)
        E = np.vstack(o.E)

        te = np.hstack(o.t[::2])
        xe = np.vstack(o.x[::2])
        ye = np.vstack(o.y[::2])
        thetae = np.vstack(o.theta[::2])

        z = np.array([v[-1],delta[-1],self.theta[-1],dtheta[-1]])

        r = 1.01

        mx,Mx,dx = (self.x.min(),self.x.max(),self.x.max()-self.x.min())
        my,My,dy = (self.y.min(),self.y.max(),self.y.max()-self.y.min())
        dd = 5*r

        self.fig = plt.figure(DoublePlot.plotNumberInc(),figsize=(5*(Mx-mx+2*dd)/(My-my+2*dd),5))
        plt.clf()
        ax = self.fig.add_subplot(111,aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

        self.Lcom, = ax.plot(self.x[0], self.y[0], 'b.', ms=10.)
        self.Ecom, = ax.plot(*DoublePlot.Ellipse((self.x[0],self.y[0]), (r, 0.5*r), t=self.theta[0]))
        self.Ecom.set_linewidth(4.0)
        self.Lft,  = ax.plot([self.x[0],self.fx[0]],[self.y[0],self.fy[0]],'g.-',lw=4.)

        ax.set_xlim((mx-dd,Mx+dd))
        ax.set_ylim((my-dd,My+dd))


    def animIterable(self,i=None):
        """
        .animIterable  animates trajectory one point at a time

        INPUTS:
            i - index to animate
        """
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

        self.Lcom.set_xdata(self.x[i])
        self.Lcom.set_ydata(self.y[i])
        self.Lft.set_xdata([self.x[i],self.fx[i]])
        self.Lft.set_ydata([self.y[i],self.fy[i]])
        Ex,Ey = DoublePlot.Ellipse((self.x[i],self.y[i]), (0.5*self.r, self.r), t=self.theta[i])
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
