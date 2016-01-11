from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import scipy.optimize as spo
import numpy as np
import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import os
import pickle
import scipy
from scipy import signal
from scipy import interpolate
from scipy import integrate
import pandas as pd
import math
from collections import deque
import copy
import random
import footlocation as fl
from shrevz import util as shutil
import pwhamil as pw
from mpl_toolkits.mplot3d import Axes3D

class Cockroach(object):
    def __init__(self, Q, P, T, F):
        '''
        Inputs:
            Q = [x, y, theta, xdot, ydot, thetadot]
            P = [x_cl, y_cl, x_cr, x_cl]
            T = [t_0, t_1, h, gaitLength]
            F = [f_l, f_r]
        Outputs:
            Y = [t, x, y, theta, xdot, ydot, thetadot, x_fl, y_fl]
        '''
        #states
        self.Q = Q
        #parameters
        self.x_cl = P[0]
        self.y_cl = P[1]
        self.x_cr = P[2]
        self.y_cr = P[3]
        self.theta_l = P[4]
        self.theta_r = P[5]
        #time info
        self.t_i = 0
        self.t_0 = T[0]
        self.t_1 = T[1]
        self.h = T[2]
        self.gaitTime = T[3]
        self.gait_i = 0
        self.stanceIndicator = True
        #output
        self.Y = np.zeros((np.shape(q)[0]+3, (self.t_1-self.t_0)/self.h+1.1))
        self.Y[:,0] = np.hstack((self.t_0, self.Q, np.asarray([self.x_cl, self.y_cl])))
        #functions
        self.f_l = F[0]
        self.f_r = F[1]
        #cockroach frame
        self.x_c = 0.0
        self.y_c = 0.0

        self.t_i = 0
        if self.stanceIndicator == True:
            self.Q[0] += self.x_cl
            self.Q[1] += self.y_cl
            self.Q[2] = self.theta_l
        elif self.stanceIndicator == False:
            self.Q[0] += self.x_cr
            self.Q[1] += self.y_cr
            self.Q[2] = self.theta_r

        #Set angle to new frame
        self.theta_c = 0.0
        #self.theta_c = q[2]
        #q[2] = 0.000

    def output(self, q):
        q = copy.deepcopy(q)
        #print 'x:' + str(q[0])
        #print 'y:' + str(q[1])

        #print np.asarray([self.x_c, self.y_c])
        q, fl = self.centerFrame(q)
        q, fl = self.toOrigin(q, fl)
        #print 'end q'
        #print q[0:3]
        tq = np.concatenate((np.asarray([self.t_i*self.h]), q, fl))
        self.Y[:,self.t_i] = tq
        return self.Y

    #Fix Theta
    def toOrigin(self, q, fl):
        q[0:2] = np.dot(rotMatrix2(self.theta_c),q[0:2]) + np.asarray([self.x_c, self.y_c])
        q[3:5] = np.dot(rotMatrix2(self.theta_c),q[3:5])
        fl = np.dot(rotMatrix2(self.theta_c),fl) + np.asarray([self.x_c, self.y_c])
        q[2] += self.theta_c
        return q, fl

    def centerFrame(self, q):
        fl = np.zeros(2)
        if self.stanceIndicator == True:
            q[0] = q[0] - self.x_cl
            q[1] = q[1] - self.y_cl
            q[2] = q[2] - self.theta_l
            fl[0] = self.x_cl
            fl[1] = self.y_cl

        elif self.stanceIndicator == False:
            q[0] = q[0] - self.x_cr
            q[1] = q[1] - self.y_cr
            q[2] = q[2] - self.theta_r
            fl[0] = self.x_cr
            fl[1] = self.y_cr


        return q, fl

    def sim(self):
        #Initialize with left foot down
        #self.initializeSim()

        while self.t_i * self.h < self.t_1:
            self.t_i += 1
            #self.Q[2] = max(min(.01, self.Q[2]), -.01)
            self.Q = self.Q + self.h*self.f()

            self.output(self.Q)
            self.checkFrame()

    def checkFrame(self):
        self.gait_i += 1
        if self.gait_i * self.h >= self.gaitTime:
            self.newFrame()
            self.gait_i = 0

    '''
    def initializeSim(self):
        self.t_i = 0
        if self.stanceIndicator == True:
            self.Q[0] += self.x_cl
            self.Q[1] += self.y_cl
        elif self.stanceIndicator == False:
            self.Q[0] += self.x_cr
            self.Q[1] += self.y_cr
        self.theta_c = q[2]
        q[2] = 0.000
    '''

    def newFrame(self):
        #Remove leg offset
        print 'NEW FRAME'
        print 'x: ' + str(self.Q[0])
        print 'y: ' + str(self.Q[1])
        print

        '''
        if self.stanceIndicator == True:
            self.x_c -= np.dot(rotMatrix2(-(self.theta_c)), np.asarray([self.x_cl, self.y_cl]))[0]
            self.y_c -= np.dot(rotMatrix2(-(self.theta_c)), np.asarray([self.x_cl, self.y_cl]))[1]
        elif self.stanceIndicator == False:
            self.x_c -= np.dot(rotMatrix2(-(self.theta_c)), np.asarray([self.x_cr, self.y_cr]))[0]
            self.y_c -= np.dot(rotMatrix2(-(self.theta_c)), np.asarray([self.x_cr, self.y_cr]))[1]
        '''
        self.centerFrame(self.Q)
        self.stanceIndicator = not self.stanceIndicator

        self.x_c += np.dot(rotMatrix2(self.theta_c), self.Q[0:2])[0]
        self.y_c += np.dot(rotMatrix2(self.theta_c), self.Q[0:2])[1]
        #self.Q[3:5] = np.dot(rotMatrix2((self.theta_c)), self.Q[3:5])

        #Set theta to zero
        self.Q[3:5] = np.dot(rotMatrix2(-self.Q[2]), self.Q[3:5])
        self.theta_c += self.Q[2]
        self.Q[2] = 0.0

        if self.stanceIndicator == True:
            self.Q[0] = self.x_cl
            self.Q[1] = self.y_cl
            self.Q[2] = self.theta_l
        elif self.stanceIndicator == False:
            self.Q[0] = self.x_cr
            self.Q[1] = self.y_cr
            self.Q[2] = self.theta_r

        print 'x: ' + str(self.x_c)
        print 'y: ' + str(self.y_c)
        print

    def f(self):
        dQ = np.zeros(6)
        dQ[0:3] = self.Q[3:6]
        if self.stanceIndicator == True:
            dQ[3:6] = self.f_l(self.Q[0:3][np.newaxis])[0,:]
        elif self.stanceIndicator == False:
            dQ[3:6] = self.f_r(self.Q[0:3][np.newaxis])[0,:]
        return dQ

def rotMatrix2(theta):
    rotMatrix = np.asarray([
    [np.cos(theta), -np.sin(theta)]
    ,[np.sin(theta), np.cos(theta)]
    ])
    return rotMatrix

def vectorField(figNum, plotTitle, f, theta):
    delta = 1.0
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z_x = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    Z_y = np.zeros((np.shape(X)[0], np.shape(X)[1]))

    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            Z_x[i,j] = f(np.asarray([[X[i,j], Y[i,j], theta]]))[0,0]
            Z_y[i,j] = f(np.asarray([[X[i,j], Y[i,j], theta]]))[0,1]

    M = np.sqrt(Z_x**2 + Z_y**2)

    plt.figure(figNum,figsize=(12,8))
    plt.clf()
    plt.quiver(X, Y, Z_x, Z_y, pivot='middle', headwidth=4, headlength=6)
    plt.xlabel('x (cm)', fontsize=16)
    plt.ylabel('y (cm)', fontsize=16)
    plt.title('Linear Acceleration Gradient, No Angle', fontsize=18)
    plt.savefig('2016SICB/plots/VectorField-'+plotTitle+'.png', bbox_inches='tight', dpi=200)
    #plt.show()

def potential3D(figNum, plotTitle, f, theta):
    delta = 0.1
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    Z_x = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    Z_y = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    #Z_mag = np.zeros((np.shape(X)[0], np.shape(X)[1]))

    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            Z_x[i,j] = f(np.asarray([[X[i,j], Y[i,j], theta]]))[0,0]
            Z_y[i,j] = f(np.asarray([[X[i,j], Y[i,j], theta]]))[0,1]

    M = np.sqrt(Z_x**2 + Z_y**2)

    fig = plt.figure(figNum,figsize=(12,8))
    ax = fig.gca(projection='3d');
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    surf = ax.plot_surface(X, Y, M, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.0, antialiased=False)
    ax.set_xlim((-5, 5))
    plt.xlabel('x (cm)', fontsize=16)
    plt.ylabel('y (cm)', fontsize=16)
    plt.title('Linear Acceleration Magnitude, Zero Angle', fontsize=18)
    #ax.set_zlabel('y (cm)')

    fig.colorbar(surf, shrink=0.5, aspect=25)
    plt.savefig('2016SICB/plots/3DPotential-'+plotTitle+'.png', bbox_inches='tight', dpi=200)

def flipFunction(lList, pwHamilResults, qRight):
    qRight = np.asarray([[qRight[0,0], -qRight[0,1], -qRight[0,2]]])
    a = pw.evaluateGradientFunction(lList, np.asarray([pwHamilResults['cLeft']]).T, qRight)
    a = -np.asarray([[a[0,0], -a[0,1], -a[0,2]]])
    return a

if __name__ == "__main__":
    saveDir = 'StableOrbit'

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)

    pwHamilResults = pw.loadPWHamil('inertia-raw')
    lList = pw.gradientFunctionsToLambdas(pwHamilResults['dV'], pwHamilResults['variables'])
    f_l = lambda qLeft : pw.evaluateGradientFunction(lList, np.asarray([pwHamilResults['cLeft']]).T, qLeft)
    #f_l = lambda _ : np.asarray([[0, 0, 0]])
    #f_r = lambda qRight : pw.evaluateGradientFunction(lList, np.asarray([pwHamilResults['cRight']]).T, qRight)
    '''
    cRight = copy.deepcopy(pwHamilResults['cLeft'])
    cRight[0] = -cRight[0]
    cRight[1] = -cRight[1]
    cRight[2] = -cRight[2]
    cRight[3] = -cRight[3]
    cRight[4] = -cRight[4]
    cRight[5] = -cRight[5]
    cRight[6] = -cRight[6]
    cRight[7] = -cRight[7]
    cRight[8] = -cRight[8]
    print cRight
    print pwHamilResults['cLeft']
    '''
    qFlip = lambda q : np.asarray([[q[0, 0]], [-q[0, 1]], [-q[0, 2]]])
    #f_r = lambda qRight : pw.evaluateGradientFunction(lList, np.asarray([pwHamilResults['cRight']]).T, qRight)
    f_r = lambda qRight : flipFunction(lList, pwHamilResults, qRight)

    test = lambda _ : np.asarray([[0, 0, 0]])

    #Q = [x, y, theta, xdot, ydot, thetadot]
    #q = np.asarray([0., 0., 0., 32.5, 0, 0.310])
    q = np.asarray([0., 0., 0.0, 32.5, 0.0, 0.0])
    p = np.asarray([2.8, 2.5, 2.8, -2.5, 0.0, 0.0])
    t = np.asarray([0.0, 0.299, 1.0e-5, 0.05])
    test = Cockroach(q, p, t, [f_l,f_r])
    test.sim()

    y = test.Y

    plt.figure(1)
    plt.clf()
    plt.plot(y[1,:],y[2,:],'b.-')
    plt.title('x v y')

    plt.plot(y[7,:],y[8,:],'ro')
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.plot(y[0,:],y[1,:],'r.-')
    plt.title('x')
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.plot(y[0,:],y[2,:],'r.-')
    plt.title('y')
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(y[0,:],y[3,:],'r.-')
    plt.title('theta')
    plt.show()

    plt.figure(5)
    plt.clf()
    plt.plot(y[0,:],y[4,:],'r.-')
    plt.title('xdot')
    plt.show()

    plt.figure(6)
    plt.clf()
    plt.plot(y[0,:],y[5,:],'r.-')
    plt.title('ydot')
    plt.show()

    plt.figure(7)
    plt.clf()
    plt.plot(y[0,:],y[6,:],'r.-')
    plt.title('thetadot')
    plt.show()

    '''
    theta = -0.0
    vectorField(4, 'leftFoot', f_l, theta)
    potential3D(5, 'leftFoot', f_l, theta)
    plt.show()
    '''
