from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import scipy.optimize as spo
import numpy as np
import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import pickle
import scipy
from scipy import signal
from scipy import interpolate
import pandas as pd
import math
from collections import deque
import copy
import random
import footlocation as fl
from shrevz import util as shutil
import multiprocessing as mp

saveDir = 'StableOrbit'

varList = ['x','y','theta']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

GLOBAL_NOISE = 0.001**2
GRID_SAMPLES = 64
ORDER = 3

def modelAccel(observation, template):
    observation['accx'] = 0
    observation['accy'] = 0
    observation['acctheta'] = 0

    for i in observation.index:
        theta = observation.ix[i, 'theta'] + np.pi/2
        x = observation.ix[i, 'x']
        y = observation.ix[i, 'y']
        fx = observation.ix[i, 'fx']
        fy = observation.ix[i, 'fy']

        f = np.array([fx,fy])
        c = np.array([x,y])

        h = c + template['d'] * np.array([np.sin(theta),np.cos(theta)])
        # leg length
        eta = np.linalg.norm(h - f)
        dV = template['k'] * (eta - template['eta0'])

        # Cartesian dynamics
        dx = [observation.ix[i, 'dx'], observation.ix[i, 'dy'], observation.ix[i, 'dtheta'],
            -dV*(x + template['d']*np.sin(theta) - fx)/(template['m']*eta),
            -dV*(y + template['d']*np.cos(theta) - fy)/(template['m']*eta),
            -dV*template['d']*((x - fx)*np.cos(theta)
            - (y - fy)*np.sin(theta))/(template['I']*eta)]

        observation.ix[i, 'accx'] = dx[3]
        observation.ix[i, 'accy'] = dx[4]
        observation.ix[i, 'acctheta'] = dx[5]

    return observation

def kalman_model_x(template, xDim, zDim, X_):
    f = KalmanFilter(dim_x=xDim, dim_z=zDim)

    dt = template['dt']

    #not worth tweaking
    #measurement noise (cm)
    f.R = np.array([[0.1]])
    f.R = .5*(f.R + f.R.T)

    #initial state estimate (cm, cm/sec, cm/sec**2)
    f.x = np.array([X_[0], np.diff(X_[:2])/dt, 0.0])

    #initial covariance estimate
    f.P = np.array([[f.R[0,0], 0.0,              0.0], \
                    [0.0,      template['v']/1, 0.0], \
                    [0.0,      0.0,              0.5]])
    f.P = .5*(f.P + f.P.T)

    # worth tweaking:
    #process noise
    f.Q = np.array([[template['v']*dt/1000, 0.0, 0.0], \
                    [0.0,                   .0001, 0.0], \
                    [0.0,                   0.0, 10000.0]])
    f.Q = .5*(f.Q + f.Q.T)

    #state transition
    veps = 0e-2
    aeps = 0e-2
    f.F = np.array([[1.0, dt,  .5*dt**2], \
                    [0.0, 1.0-veps, dt], \
                    [0.0, 0.0, 1.0-aeps]])

    #readable attributes
    f.H = np.array([[1.0, 0.0, 0.0]])

    return f

def kalman_data_x(template, xDim, zDim, X_):
    f = KalmanFilter(dim_x=xDim, dim_z=zDim)

    dt = template['dt']

    #not worth tweaking
    #measurement noise (cm)
    f.R = np.array([[0.1]])
    f.R = .5*(f.R + f.R.T)

    #initial state estimate (cm, cm/sec, cm/sec**2)
    f.x = np.array([X_[0], np.diff(X_[:2])/dt, 0.0])

    #initial covariance estimate
    f.P = np.array([[f.R[0,0], 0.0,              0.0], \
                    [0.0,      template['v']/1, 0.0], \
                    [0.0,      0.0,              0.5]])
    f.P = .5*(f.P + f.P.T)

    # worth tweaking:
    #process noise
    f.Q = np.array([[template['v']*dt, 0.0, 0.0], \
                    [0.0,                   1000.0, 0.0], \
                    [0.0,                   0.0, 1000.0]])
    f.Q = .5*(f.Q + f.Q.T)

    #state transition
    veps = 1e-2
    aeps = 1e-2
    f.F = np.array([[1.0, dt,  .5*dt**2], \
                    [0.0, 1.0-veps, dt], \
                    [0.0, 0.0, 1.0-aeps]])

    #readable attributes
    f.H = np.array([[1.0, 0.0, 0.0]])

    return f

'''
def kalman_tune_x(template, xDim, zDim, X_, decVar):
    f = KalmanFilter(dim_x=xDim, dim_z=zDim)

    dt = template['dt']

    #not worth tweaking
    #measurement noise (cm)
    f.R = np.array([[0.1]])
    f.R = .5*(f.R + f.R.T)

    #initial state estimate (cm, cm/sec, cm/sec**2)
    f.x = np.array([X_[0], np.diff(X_[:2])/dt, 0.0])

    #initial covariance estimate
    f.P = np.array([[f.R[0,0], 0.0,              0.0], \
                    [0.0,      template['v']/1, 0.0], \
                    [0.0,      0.0,              0.5]])
    f.P = .5*(f.P + f.P.T)

    # worth tweaking:
    #process noise
    f.Q = np.array([[decVar[0], decVar[1], decVar[2]], \
                    [0.0,       decVar[3], decVar[4]], \
                    [0.0,             0.0, decVar[5]]])
    f.Q = .5*(f.Q + f.Q.T)

    #state transition
    veps = 1e-2
    aeps = 1e-2
    f.F = np.array([[1.0, dt,  .5*dt**2], \
                    [0.0, 1.0-veps, dt], \
                    [0.0, 0.0, 1.0-aeps]])

    #readable attributes
    f.H = np.array([[1.0, 0.0, 0.0]])

    return f

'''
def kalman_tune_x(template, xDim, zDim, X_, decVar):
    f = KalmanFilter(dim_x=xDim, dim_z=zDim)

    dt = template['dt']

    #not worth tweaking
    #measurement noise (cm)
    f.R = np.array([[GLOBAL_NOISE]])
    f.R = .5*(f.R + f.R.T)

    #initial state estimate (cm, cm/sec, cm/sec**2)
    #f.x = np.array([X_[0], np.diff(X_[:2])/dt, 0.0])
    f.x = np.array([X_[0], np.median(np.diff(X_[:90])/dt), 0.0])

    #initial covariance estimate
    f.P = np.array([[f.R[0,0], 0.0,              0.0], \
                    [0.0,      template['v']/1, 0.0], \
                    [0.0,      0.0,              0.5]])
    f.P = .5*(f.P + f.P.T)

    # worth tweaking:
    #process noise
    f.Q = np.array([[decVar[0], decVar[1], decVar[2]], \
                    [0.0,       decVar[3], decVar[4]], \
                    [0.0,             0.0, decVar[5]]])
    f.Q = .5*(f.Q + f.Q.T)

    #state transition
    veps = 0#1e-2
    aeps = 0#1e-2
    f.F = np.array([[1.0, dt,  .5*dt**2], \
                    [0.0, 1.0-veps, dt], \
                    [0.0, 0.0, 1.0-aeps]])

    #readable attributes
    f.H = np.array([[1.0, 0.0, 0.0]])

    return f

def kalman_tune_x2(template, xDim, zDim, X_, decVar):
    f = KalmanFilter(dim_x=xDim, dim_z=zDim)

    dt = template['dt']

    #not worth tweaking
    #measurement noise (cm)
    f.R = np.array([[GLOBAL_NOISE]])
    f.R = .5*(f.R + f.R.T)

    #initial state estimate (cm, cm/sec, cm/sec**2)
    #f.x = np.array([X_[0], np.diff(X_[:2])/dt, 0.0])
    f.x = np.array([X_[0], np.median(np.diff(X_[:90])/dt), 0.0])

    #initial covariance estimate
    f.P = np.array([[f.R[0,0],             0.0, 0.0], \
                    [     0.0, template['v']/1, 0.0], \
                    [     0.0,             0.0, 0.5]])
    f.P = .5*(f.P + f.P.T)

    # worth tweaking:
    #process noise
    f.Q = np.array([[decVar[0],       0.0,       0.0], \
                    [0.0,       decVar[1],       0.0], \
                    [0.0,             0.0, decVar[2]]])
    f.Q = .5*(f.Q + f.Q.T)

    #state transition
    veps = 0#1e-2
    aeps = 0#1e-2
    f.F = np.array([[1.0, dt,  .5*dt**2], \
                    [0.0, 1.0-veps, dt], \
                    [0.0, 0.0, 1.0-aeps]])

    #readable attributes
    f.H = np.array([[1.0, 0.0, 0.0]])

    return f

def modelSimulate(X_, kalmanFilter, kt, decVar):
    template = mc.jsonLoadTemplate('templateControl')

    xDim = 3
    zDim = 1

    if decVar != None:
        f = eval('kalman_' + kalmanFilter)(template, xDim, zDim, X_, decVar)
    else:
        f = eval('kalman_' + kalmanFilter)(template, xDim, zDim, X_)

    f2 = copy.deepcopy(f)

    #signal = [t + 1 for t in range (40)]
    #signal = mw.observations[0]['x'].dropna()

    # corresponds to error of 1% body length
    #np.random.seed(seed=2)
    #noise = np.sqrt(GLOBAL_NOISE)*np.random.randn(*X_.shape)
    #signal = X_ + noise
    signal = kt.X_

    (mu1, cov1, _, _) = f.batch_filter(signal)
    (x1, P1, k1) = f.rts_smoother(mu1, cov1)

    #np.random.seed(seed=3)
    #noise = np.sqrt(GLOBAL_NOISE)*np.random.randn(*X_.shape)
    #signal = X_ + noise
    signal2 = kt.X_2

    (mu2, cov2, _, _) = f2.batch_filter(signal2)
    (x2, P2, k2) = f2.rts_smoother(mu2, cov2)

    return (x1, P1, x2, P2, signal, f)

def plotModelSimulate(template, var, kalmanFilter, kt, decVar):
    observation = modelAccel(mw.observations[0], template)

    X_ = np.asarray(kt.X_)
    dX_ = np.asarray(kt.dX_)
    ddX_ = np.asarray(observation['acc' + var])

    (x1, P1, x2, P2, signal, f) = modelSimulate(X_, kalmanFilter, kt, decVar)

    #X,dX,ddX = mu.T
    Xs,dXs,ddXs = x1.T
    Xsr,dXsr,ddXsr = x2.T

    # plots
    xlim = (0,600)

    plt.figure(num=1,figsize=(10, 10)); plt.clf()
    ax = plt.subplot(3,1,1); ax.grid('on')
    ax.plot(X_,'.',lw=4.,color='gray')
    #ax.plot(X, 'b.-',lw=2.)
    ax.plot(Xs,'g.-',lw=1.)
    ax.plot(Xsr,'b.-',lw=1.)
    ax.plot(signal,'k.',lw=1.)
    ax.set_xlim(xlim)
    ax.set_ylabel('pos (cm)')
    ax = plt.subplot(3,1,2); ax.grid('on')
    ax.plot(dX_,'.',lw=4.,color='gray')
    #ax.plot(np.diff(signal)/dt,'.',lw=4.,color='red')
    #ax.plot(dX, 'b.-',lw=4.)
    ax.plot(dXs,'g.-',lw=2.)
    ax.plot(dXsr,'b.-',lw=2.)
    ax.set_xlim(xlim)
    ax.set_ylabel('vel (cm/sec)')
    ax = plt.subplot(3,1,3); ax.grid('on')
    ax.plot(ddX_,'.',lw=4.,color='gray')
    #ax.plot(np.diff(signal)/dt,'.',lw=4.,color='red')
    #ax.plot(dX, 'b.-',lw=4.)
    ax.plot(ddXs,'g.-',lw=2.)
    ax.plot(ddXsr,'b.-',lw=2.)
    ax.set_xlim(xlim)
    ax.set_ylabel('acc (cm/sec**2)')
    ax.set_xlabel('time (samp)')
    plt.show()

    return (x1, P1, x2, P2, f)

def dataKalman(X_, kalmanFilter, decVar):
    template = mc.jsonLoadTemplate('templateControl')

    xDim = 3
    zDim = 1

    if decVar != None:
        f = eval('kalman_' + kalmanFilter)(template, xDim, zDim, X_, decVar)
    else:
        f = eval('kalman_' + kalmanFilter)(template, xDim, zDim, X_)
    #signal = [t + 1 for t in range (40)]
    #signal = mw.observations[0]['x'].dropna()

    # corresponds to error of 1% body length
    signal = X_

    (mu, cov, _, _) = f.batch_filter(signal)
    (x, P, k) = f.rts_smoother(mu, cov)

    return (mu, cov, x, P)

def plotDataKalman(data, dataIDs, column, kalmanFilter, kt, decVar = None):
    columnX = column + '_KalmanX'
    columndX = column + '_KalmanDX'
    columnddX = column + '_KalmanDDX'
    for dataID in dataIDs:
        currentData = data[dataID]
        indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(currentData, 283, 'pre')
        X_ = np.asarray(currentData[column][0:283].dropna())
        (mu, cov, x, P) = dataKalman(X_, kalmanFilter, decVar)
        currentData[columnX] = float('nan')
        currentData[columndX] = float('nan')
        currentData[columnddX] = float('nan')
        currentData[columnX][indexStart:indexEnd] = x[:,0]
        currentData[columndX][indexStart:indexEnd] = x[:,1]
        currentData[columnddX][indexStart:indexEnd] = x[:,2]

    (phaseGrid_dx, phaseMean_dx, phaseLQuantile_dx, phaseUQuantile_dx) = fl.findPhaseValues(data, dataIDs, columndX)
    (phaseGrid_ddx, phaseMean_ddx, phaseLQuantile_ddx, phaseUQuantile_ddx) = fl.findPhaseValues(data, dataIDs, columnddX)

    xlim = 2*np.pi
    plt.figure(num=2,figsize=(10, 10)); plt.clf()
    ax = plt.subplot(2,1,1); ax.grid('on')
    ax.plot(phaseGrid_dx, phaseMean_dx,'g-', lw=3.)
    ax.plot(phaseGrid_dx, phaseLQuantile_dx,'g--',lw=2.)
    ax.plot(phaseGrid_dx, phaseUQuantile_dx,'g--',lw=2.)
    ax.set_xlim(0,xlim)
    ax.set_ylabel('velocity (cm/s)')
    ax = plt.subplot(2,1,2); ax.grid('on')
    ax.plot(phaseGrid_ddx, phaseMean_ddx,'g-', lw=3.)
    ax.plot(phaseGrid_ddx, phaseUQuantile_ddx,'g--',lw=2.)
    ax.plot(phaseGrid_ddx, phaseLQuantile_ddx,'g--',lw=2.)
    ax.set_xlim(0,xlim)
    ax.set_ylabel('acceleration (cm/s**2)')
    ax.set_xlabel('phase (radians)')
    plt.show()

def columnTableX(column):
    columnX = column + '_KalmanX'
    columndX = column + '_KalmanDX'
    columnddX = column + '_KalmanDDX'
    return (columnX, columndX, columnddX)

def kalmanFilterData(data, dataIDs, column, kalmanFilter, kt, decVar = None):
    (columnX, columndX, columnddX) = columnTableX(column)
    for dataID in dataIDs:
        currentData = data[dataID]
        indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(currentData, 284, 'pre')
        X_ = np.asarray(currentData[column][0:284].dropna())
        (mu, cov, x, P) = dataKalman(X_, kalmanFilter, decVar)
        currentData[columnX] = float('nan')
        currentData[columndX] = float('nan')
        currentData[columnddX] = float('nan')
        currentData[columnX][indexStart:indexEnd] = x[:,0]
        currentData[columndX][indexStart:indexEnd] = x[:,1]
        currentData[columnddX][indexStart:indexEnd] = x[:,2]
    return data

def kalmanBootstrap(data, dataIDs, samples, column):
    random.seed(2)
    means = {}
    (columnX, columndX, columnddX) = columnTableX(column)
    for i in range(samples):
        randomDataIDs = np.random.choice(dataIDs, len(dataIDs))
        mean = pd.DataFrame(columns=[columnX, columndX, columnddX], index=range(GRID_SAMPLES))
        mean[columnX] = 0.0
        mean[columndX] = 0.0
        mean[columnddX] = 0.0
        for dataID in dataIDs:
            (phaseGrid_x, phaseMean_x, phaseLQuantile_x, phaseUQuantile_x) = fl.findPhaseValues(data, [dataID], columnX)
            (phaseGrid_dx, phaseMean_dx, phaseLQuantile_dx, phaseUQuantile_dx) = fl.findPhaseValues(data, [dataID], columndX)
            (phaseGrid_ddx, phaseMean_ddx, phaseLQuantile_ddx, phaseUQuantile_ddx) = fl.findPhaseValues(data, [dataID], columnddX)
            mean[columnX] += sum(randomDataIDs == dataID) * pd.Series(phaseMean_x).fillna(0.0)
            mean[columndX] += sum(randomDataIDs == dataID) * pd.Series(phaseMean_dx).fillna(0.0)
            mean[columnddX] += sum(randomDataIDs == dataID) * pd.Series(phaseMean_ddx).fillna(0.0)
        #mean[mean == 0.0] = np.nan
        mean = mean/len(dataIDs)
        means[i] = mean
    return means

def kalmanBootstrapShai(data, dataIDs, samples, column):
    random.seed(2)
    means = {}
    (columnX, columndX, columnddX) = columnTableX(column)
    for i in range(samples):
        randomDataIDs = np.random.choice(dataIDs, len(dataIDs))
        mean = pd.DataFrame(columns=[columnX, columndX, columnddX], index=range(GRID_SAMPLES))
        mean[columnX] = 0.0
        mean[columndX] = 0.0
        mean[columnddX] = 0.0
        phase = np.array([[]])
        valuesX = np.array([[]])
        valuesdX = np.array([[]])
        valuesddX = np.array([[]])
        for dataID in randomDataIDs:
            phase = np.hstack((phase, data[dataID]['Roach_xv_phase'][0:283].dropna()[np.newaxis,:]))
            valuesX = np.hstack((valuesX, data[dataID][columnX][0:283].dropna()[np.newaxis,:]))
            valuesdX = np.hstack((valuesdX, data[dataID][columndX][0:283].dropna()[np.newaxis,:]))
            valuesddX = np.hstack((valuesddX, data[dataID][columnddX][0:283].dropna()[np.newaxis,:]))
        fsX = shutil.FourierSeries()
        fsdX = shutil.FourierSeries()
        fsddX = shutil.FourierSeries()
        fsX.fit(ORDER, copy.deepcopy(phase), valuesX)
        fsdX.fit(ORDER, copy.deepcopy(phase), valuesdX)
        fsddX.fit(ORDER, copy.deepcopy(phase), valuesddX)
        grid  = np.arange(GRID_SAMPLES)[np.newaxis,:] * (2*np.pi)/GRID_SAMPLES
        mean[columnX] = fsX.val(grid)[:,0].real
        mean[columndX] = fsdX.val(grid)[:,0].real
        mean[columnddX] = fsddX.val(grid)[:,0].real
        #mean = mean/len(dataIDs)
        means[i] = mean
    return means

def kalmanMeanStats(means, column):
    meansList_x = {}
    meansList_dx = {}
    meansList_ddx = {}
    (columnX, columndX, columnddX) = columnTableX(column)
    for i in range(GRID_SAMPLES):
        meansList_x[i] = deque()
        meansList_dx[i] = deque()
        meansList_ddx[i] = deque()

    for meanID in means.keys():
        for i in range(GRID_SAMPLES):
            meansList_x[i].append(means[meanID].ix[i, columnX])
            meansList_dx[i].append(means[meanID].ix[i, columndX])
            meansList_ddx[i].append(means[meanID].ix[i, columnddX])


    columns=[columnX + '_Mean', columnX + '_01', columnX + '_99' \
    ,columndX + '_Mean', columndX + '_01', columndX + '_99' \
    ,columnddX + '_Mean', columnddX + '_01', columnddX + '_99']
    meanStats = pd.DataFrame(columns=columns, index=range(GRID_SAMPLES))

    for i in range(GRID_SAMPLES):
        xList = pd.Series(meansList_x[i])
        dxList = pd.Series(meansList_dx[i])
        ddxList = pd.Series(meansList_ddx[i])
        if all(np.isnan(a) for a in xList):
            continue
        meanStats.ix[i, columnX + '_Mean'] = xList.mean()
        meanStats.ix[i, columndX + '_Mean'] = dxList.mean()
        meanStats.ix[i, columnddX + '_Mean'] = ddxList.mean()
        meanStats.ix[i, columnX + '_01'] = xList.quantile(q=[.01]).ix[.01]
        meanStats.ix[i, columndX + '_99'] = dxList.quantile(q=[.01]).ix[.01]
        meanStats.ix[i, columnddX + '_01'] = ddxList.quantile(q=[.01]).ix[.01]
        meanStats.ix[i, columnX + '_99'] = xList.quantile(q=[.99]).ix[.99]
        meanStats.ix[i, columndX + '_01'] = dxList.quantile(q=[.99]).ix[.99]
        meanStats.ix[i, columnddX + '_99'] = ddxList.quantile(q=[.99]).ix[.99]

    return meanStats

def kalmanDataPlotMeanStats(meanStats, column):
    (columnX, columndX, columnddX) = columnTableX(column)
    xlim = 2*np.pi
    phaseGrid = meanStats.index * (2 * np.pi)/64
    plt.figure(num=2,figsize=(10, 10)); plt.clf()
    ax = plt.subplot(2,1,1); ax.grid('on')
    ax.plot(phaseGrid, meanStats[columndX + '_Mean'],'g.', lw=3.)
    ax.plot(phaseGrid, meanStats[columndX + '_01'],'g-',lw=2.)
    ax.plot(phaseGrid, meanStats[columndX + '_99'],'g-',lw=2.)
    ax.set_xlim(0,xlim)
    ax.set_ylabel('velocity (cm/s)')
    ax = plt.subplot(2,1,2); ax.grid('on')
    ax.plot(phaseGrid, meanStats[columnddX + '_Mean'],'g.', lw=3.)
    ax.plot(phaseGrid, meanStats[columnddX + '_01'],'g-',lw=2.)
    ax.plot(phaseGrid, meanStats[columnddX + '_99'],'g-',lw=2.)
    ax.set_xlim(0,xlim)
    ax.set_ylabel('acceleration (cm/s**2)')
    ax.set_xlabel('phase (radians)')
    plt.show()

def bootstrapDataKalman(data, dataIDs, samples, column, kalmanFilter, kt, decVar = None):
    data = kalmanFilterData(data, dataIDs, column, kalmanFilter, kt, decVar)
    means = kalmanBootstrapShai(data, dataIDs, samples, column)
    meanStats = kalmanMeanStats(means, column)
    return meanStats

def kalmanDataFigures(saveLabel, color, meanStatsX, meanStatsY, meanStatsTheta, meanStatsPitch, meanStatsRoll):
    period = 2*np.pi
    phaseGrid = meanStatsX.index * (2 * np.pi)/64

    saveLabel = str(saveLabel)

    (columnX, columndX, columnddX) = columnTableX('Roach_x')
    plt.figure(1,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsX[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsX[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsX[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x position(cm)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-1.svg')

    plt.figure(2,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsX[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsX[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsX[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x velocity (cm/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-2.svg')

    plt.figure(3,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsX[columnddX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsX[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsX[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x acceleration (cm/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-3.svg')

    (columnX, columndX, columnddX) = columnTableX('Roach_y')
    plt.figure(4,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsY[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsY[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsY[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('y position (cm)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-4.svg')

    plt.figure(5,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsY[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsY[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsY[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('y velocity (cm/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-5.svg')

    plt.figure(6,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsY[columnddX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsY[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsY[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('y acceleration (cm/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-6.svg')

    (columnX, columndX, columnddX) = columnTableX('Roach_theta')
    plt.figure(7,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTheta[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTheta[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTheta[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular position (randians)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-7.svg')

    plt.figure(8,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTheta[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTheta[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTheta[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular velocity (radians/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-8.svg')

    plt.figure(9,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTheta[columnddX + '_Mean'],color + '.-', lw=3., markersize=12 , label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTheta[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTheta[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular acceleration (radians/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-9.svg')

    (columnX, columndX, columnddX) = columnTableX('Roach_pitch')
    plt.figure(10,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsPitch[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsPitch[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsPitch[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular position (randians)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-10.svg')

    plt.figure(11,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsPitch[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsPitch[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsPitch[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular velocity (radians/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-11.svg')

    plt.figure(12,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsPitch[columnddX + '_Mean'],color + '.-', lw=3., markersize=12 , label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsPitch[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsPitch[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular acceleration (radians/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-12.svg')

    (columnX, columndX, columnddX) = columnTableX('Roach_roll')
    plt.figure(13,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsRoll[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsRoll[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsRoll[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular position (randians)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-13.svg')

    plt.figure(14,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsRoll[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsRoll[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsRoll[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular velocity (radians/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-14.svg')

    plt.figure(15,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsRoll[columnddX + '_Mean'],color + '.-', lw=3., markersize=12 , label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsRoll[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsRoll[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('angular acceleration (radians/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-15.svg')

def kalmanDataFiguresTarsus(saveLabel, color, column, meanStatsTarsus):
    period = 2*np.pi
    phaseGrid = meanStatsTarsus.index * (2 * np.pi)/64

    saveLabel = str(saveLabel)

    (columnX, columndX, columnddX) = columnTableX(column)
    plt.figure(1,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTarsus[columnX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTarsus[columnX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTarsus[columnX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x position(cm)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-1.svg')

    plt.figure(2,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTarsus[columndX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTarsus[columndX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTarsus[columndX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x velocity (cm/s)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-2.svg')

    plt.figure(3,figsize=(12,8)); plt.clf()
    dataLegend, = plt.plot(phaseGrid, meanStatsTarsus[columnddX + '_Mean'],color + '.-', lw=3., markersize=12, label='mean')
    dataLegendS, = plt.plot(phaseGrid, meanStatsTarsus[columnddX + '_01'],color + '-',lw=2., label='98% confidence interval')
    plt.plot(phaseGrid, meanStatsTarsus[columnddX + '_99'],color + '-',lw=2.)
    plt.legend(handles=[dataLegend, dataLegendS], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,period)
    plt.ylabel('x acceleration (cm/s**2)')
    #plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull2/' + saveLabel +'-3.svg')

def dTrend(data, dataIDs, column):
    for dataID in dataIDs:
        currentData = data[dataID]
        beginIndex, endIndex = mw.findDataIndexStatic(currentData, 283, 'pre')
        endIndex += 1
        trendCoefficients = np.polyfit(x=range(len(currentData[column][beginIndex:endIndex])),y=currentData[column][beginIndex:endIndex],deg=1)
        dtrend_X = np.arange(len(currentData[column][beginIndex:endIndex])) * trendCoefficients[0] + trendCoefficients[1]
        currentData[column][beginIndex:endIndex] = currentData[column][beginIndex:endIndex] - dtrend_X
    return data, dtrend_X

def thetaCorrection(data, dataIDs):
    for dataID in dataIDs:
        currentData = data[dataID]
        currentData['Roach_theta_orig'] = np.nan
        thetaMean = currentData['Roach_theta'][0:284].mean()
        currentData['Roach_theta_orig'][0:284] = currentData['Roach_theta'][0:284]
        currentData['Roach_theta'] = currentData['Roach_theta'] - thetaMean
        beginIndex, endIndex = model.ModelWrapper.findDataIndexStatic(currentData, 283, 'pre')
        offset = np.mat([currentData.ix[beginIndex, 'Roach_x'],currentData.ix[beginIndex, 'Roach_y']])
        c = model.ModelWrapper.rotMat2(-thetaMean) * (np.mat([currentData['Roach_x'],currentData['Roach_y']]) - offset.T)
        c = c + offset.T
        currentData['Roach_x'] = pd.Series(np.squeeze(np.array(c[0,:])))
        currentData['Roach_y'] = pd.Series(np.squeeze(np.array(c[1,:])))

    return data, thetaMean

def saveLabelTrials(saveLabel, dataIDs):
    global labelTrials
    labelTrials[saveLabel] = len(dataIDs)

def generateFigs(saveLabel, color, data, dataIDs, samples, template):
    mw.csvLoadData(dataIDs)
    saveLabelTrials(saveLabel, dataIDs)
    thetaCorrection(data, dataIDs)

    ktx = KalmanTuner('tune_x2', template, 'x', [template['v']*template['dt']/50, .0001 , 1000.0])
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    data =  dTrend(data, dataIDs, 'Roach_x')
    meanStatsX = bootstrapDataKalman(mw.data, dataIDs, samples, 'Roach_x', 'tune_x2', ktx, ox[0])

    kty = KalmanTuner('tune_x2', template, 'y', [template['v']*template['dt']/100, .0001, 1000.0])
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-2, 1e-3])
    data =  dTrend(data, dataIDs, 'Roach_y')
    meanStatsY = bootstrapDataKalman(mw.data, dataIDs, samples, 'Roach_y', 'tune_x2', kty, oy[0])

    kttheta = KalmanTuner('tune_x2', template, 'theta', [template['theta']*template['dt']/50, .0001, 1000.0])
    otheta = spo.leastsq(kttheta.costx, kttheta.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    meanStatsTheta = bootstrapDataKalman(mw.data, dataIDs, samples, 'Roach_theta', 'tune_x2', kttheta, otheta[0])

    for dataID in dataIDs:
        data[dataID]['Roach_pitch'] =  data[dataID]['Roach_pitch'] * (2 * np.pi)/360
        data[dataID]['Roach_roll'] =  data[dataID]['Roach_roll'] * (2 * np.pi)/360

    meanStatsPitch = bootstrapDataKalman(mw.data, dataIDs, samples, 'Roach_pitch', 'tune_x2', ktx, ox[0])
    meanStatsRoll = bootstrapDataKalman(mw.data, dataIDs, samples, 'Roach_roll', 'tune_x2', kty, oy[0])

    #return meanStatsPitch, meanStatsRoll
    kalmanDataFigures(saveLabel, color, meanStatsX, meanStatsY, meanStatsTheta, meanStatsPitch, meanStatsRoll)

def generateFigsTarsus(label, data, dataIDs, samples, template):
    mw.csvLoadData(dataIDs)
    tarsusData = pickle.load(open( "tarsusData.p", "rb" ))

    ktx = KalmanTuner('tune_x', template, 'x', [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0])
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    meanStatsTB1x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody1_x', 'tune_x', ktx, ox[0])
    meanStatsTB2x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody2_x', 'tune_x', ktx, ox[0])
    meanStatsTB3x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody3_x', 'tune_x', ktx, ox[0])
    meanStatsTB4x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody4_x', 'tune_x', ktx, ox[0])
    meanStatsTB5x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody5_x', 'tune_x', ktx, ox[0])
    meanStatsTB6x = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody6_x', 'tune_x', ktx, ox[0])

    kty = KalmanTuner('tune_x', template, 'y', [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0])
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    meanStatsTB1y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody1_y', 'tune_x', kty, oy[0])
    meanStatsTB2y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody2_y', 'tune_x', kty, oy[0])
    meanStatsTB3y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody3_y', 'tune_x', kty, oy[0])
    meanStatsTB4y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody4_y', 'tune_x', kty, oy[0])
    meanStatsTB5y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody5_y', 'tune_x', kty, oy[0])
    meanStatsTB6y = bootstrapDataKalman(mw.data, dataIDs, samples, 'TarsusBody6_y', 'tune_x', kty, oy[0])

    #kalmanDataFiguresTarsus('tarsus1', 'b', 'TarsusBody1_x', meanStatsTB1)
    #kalmanDataFiguresTarsus('tarsus2', 'b', 'TarsusBody2_x', meanStatsTB2)
    #kalmanDataFiguresTarsus('tarsus3', 'b', 'TarsusBody3_x', meanStatsTB3)
    #kalmanDataFiguresTarsus('tarsus4', 'b', 'TarsusBody4_x', meanStatsTB4)
    #kalmanDataFiguresTarsus('tarsus5', 'b', 'TarsusBody5_x', meanStatsTB5)
    #kalmanDataFiguresTarsus('tarsus6', 'b', 'TarsusBody6_x', meanStatsTB6)

    currentData = {}
    currentData['1x'] = meanStatsTB1x
    currentData['2x'] = meanStatsTB2x
    currentData['3x'] = meanStatsTB3x
    currentData['4x'] = meanStatsTB4x
    currentData['5x'] = meanStatsTB5x
    currentData['6x'] = meanStatsTB6x
    currentData['1y'] = meanStatsTB1y
    currentData['2y'] = meanStatsTB2y
    currentData['3y'] = meanStatsTB3y
    currentData['4y'] = meanStatsTB4y
    currentData['5y'] = meanStatsTB5y
    currentData['6y'] = meanStatsTB6y
    tarsusData[label] = currentData

    pickle.dump(tarsusData, open( "tarsusData.p", "wb" ))

def dataToEstimatesTable(column, meanStats, estimatesTable):
    (columnX, columndX, columnddX) = columnTableX(column)
    estimatesTable[columnX + '_Mean'] = meanStats[columnX + '_Mean']
    estimatesTable[columnX + '_01'] = meanStats[columnX + '_01']
    estimatesTable[columnX + '_99'] = meanStats[columnX + '_99']

    estimatesTable[columndX + '_Mean'] = meanStats[columndX + '_Mean']
    estimatesTable[columndX + '_01'] = meanStats[columndX + '_01']
    estimatesTable[columndX + '_99'] = meanStats[columndX + '_99']

    estimatesTable[columnddX + '_Mean'] = meanStats[columnddX + '_Mean']
    estimatesTable[columnddX + '_01'] = meanStats[columnddX + '_01']
    estimatesTable[columnddX + '_99'] = meanStats[columnddX + '_99']

    return estimatesTable

def generateEstimateTable(label, data, dataIDs, samples, template, obsID):
    mw.csvLoadData(dataIDs)

    data, thetaMean = thetaCorrection(data, dataIDs)

    data, dtrend_X =  dTrend(data, dataIDs, 'Roach_x')
    ktx = KalmanTuner('tune_x2', template, 'x', [template['v']*template['dt']/50, .0001 , 1000.0], obsID)
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    meanStatsX = bootstrapDataKalman(data, dataIDs, samples, 'Roach_x', 'tune_x2', ktx, ox[0])

    data, dtrend_X =  dTrend(data, dataIDs, 'Roach_y')
    kty = KalmanTuner('tune_x2', template, 'y', [template['v']*template['dt']/100, .0001, 1000.0], obsID)
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-2, 1e-3])
    meanStatsY = bootstrapDataKalman(data, dataIDs, samples, 'Roach_y', 'tune_x2', kty, oy[0])

    kttheta = KalmanTuner('tune_x2', template, 'theta', [template['theta']*template['dt']/50, .0001, 1000.0], obsID)
    otheta = spo.leastsq(kttheta.costx, kttheta.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    meanStatsTheta = bootstrapDataKalman(data, dataIDs, samples, 'Roach_theta', 'tune_x2', kttheta, otheta[0])

    for dataID in dataIDs:
        data[dataID]['Roach_pitch'] =  data[dataID]['Roach_pitch'] * (2 * np.pi)/360
        data[dataID]['Roach_roll'] =  data[dataID]['Roach_roll'] * (2 * np.pi)/360

    meanStatsPitch = bootstrapDataKalman(data, dataIDs, samples, 'Roach_pitch', 'tune_x2', ktx, ox[0])
    meanStatsRoll = bootstrapDataKalman(data, dataIDs, samples, 'Roach_roll', 'tune_x2', kty, oy[0])

    ktx = KalmanTuner('tune_x', template, 'x', [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0], obsID)
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    meanStatsTB1x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody1_x', 'tune_x', ktx, ox[0])
    meanStatsTB2x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody2_x', 'tune_x', ktx, ox[0])
    meanStatsTB3x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody3_x', 'tune_x', ktx, ox[0])
    meanStatsTB4x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody4_x', 'tune_x', ktx, ox[0])
    meanStatsTB5x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody5_x', 'tune_x', ktx, ox[0])
    meanStatsTB6x = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody6_x', 'tune_x', ktx, ox[0])

    kty = KalmanTuner('tune_x', template, 'y', [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0], obsID)
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    meanStatsTB1y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody1_y', 'tune_x', kty, oy[0])
    meanStatsTB2y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody2_y', 'tune_x', kty, oy[0])
    meanStatsTB3y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody3_y', 'tune_x', kty, oy[0])
    meanStatsTB4y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody4_y', 'tune_x', kty, oy[0])
    meanStatsTB5y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody5_y', 'tune_x', kty, oy[0])
    meanStatsTB6y = bootstrapDataKalman(data, dataIDs, samples, 'TarsusBody6_y', 'tune_x', kty, oy[0])

    varList = ['Roach_x', 'Roach_y', 'Roach_theta', 'Roach_pitch', 'Roach_roll', 'TarsusBody1_x','TarsusBody2_x', \
    'TarsusBody3_x', 'TarsusBody4_x', 'TarsusBody5_x', 'TarsusBody6_x', 'TarsusBody1_y','TarsusBody2_y', \
    'TarsusBody3_y', 'TarsusBody4_y', 'TarsusBody5_y', 'TarsusBody6_y']

    estimatesTable = pd.DataFrame(index=range(64))

    estimatesTable = dataToEstimatesTable('Roach_x', meanStatsX, estimatesTable)
    estimatesTable = dataToEstimatesTable('Roach_y', meanStatsY, estimatesTable)
    estimatesTable = dataToEstimatesTable('Roach_theta', meanStatsTheta, estimatesTable)
    estimatesTable = dataToEstimatesTable('Roach_pitch', meanStatsPitch, estimatesTable)
    estimatesTable = dataToEstimatesTable('Roach_roll', meanStatsRoll, estimatesTable)

    estimatesTable = dataToEstimatesTable('TarsusBody1_x', meanStatsTB1x, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody2_x', meanStatsTB2x, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody3_x', meanStatsTB3x, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody4_x', meanStatsTB4x, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody5_x', meanStatsTB5x, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody6_x', meanStatsTB6x, estimatesTable)

    estimatesTable = dataToEstimatesTable('TarsusBody1_y', meanStatsTB1y, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody2_y', meanStatsTB2y, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody3_y', meanStatsTB3y, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody4_y', meanStatsTB4y, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody5_y', meanStatsTB5y, estimatesTable)
    estimatesTable = dataToEstimatesTable('TarsusBody6_y', meanStatsTB6y, estimatesTable)

    #estimates[label] = estimatesTable
    #pickle.dump(estimates, open( "estimates.p", "wb" ))
    pickle.dump(estimatesTable, open(os.path.join(os.getcwd(), 'estimates',label+'.p'), 'wb'))
    return estimatesTable

def checkKalmanFilter(data, dataID, template, column, var):
    ktx = KalmanTuner('tune_x', template, var, [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0])
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    data = kalmanFilterData(data, [dataID], column, 'tune_x', ktx, ox[0])
    (columnX, columndX, columnddX) = columnTableX(column)
    plt.figure(1);
    ax = plt.subplot(2,1,1); ax.grid('on')
    ax.plot(data[dataID].index[0:283], data[dataID][columnX][0:283],'g.', lw=2.)
    ax.plot(data[dataID].index[0:283], data[dataID][column][0:283],'b.',lw=2.)
    ax.set_xlabel('index')
    ax.set_ylabel('variable')
    ax = plt.subplot(2,1,2); ax.grid('on')
    ax.plot(data[dataID].index[0:283], data[dataID][columnX][0:283] - data[dataID][column][0:283],'r.', lw=2.)
    ax.set_xlabel('index')
    ax.set_ylabel('variable estimation error')
    plt.show()

def spoofKalman(template, column, var):
    random.seed(2)
    ktx = KalmanTuner('tune_x', template, var, [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0])
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    data = {}
    (columnX, columndX, columnddX) = columnTableX(column)
    data[0] = pd.DataFrame(columns=[column, columnX, columndX, columnddX],index=range(500))
    data[0].ix[0, column] = random.gauss(0, .1)
    for i in range(1,500):
        data[0].ix[i, column] = random.gauss(0, .1) + data[0].ix[i-1, column]
    data = kalmanFilterData(data, [0], column, 'tune_x', ktx, ox[0])
    return data

class KalmanTuner(object):
    #TODO: TAKE OBS ID SO YOU IT CAN USE NON CONTROL#
    def __init__(self, kalmanFilter, template, var, decVar, obsID):
        self.kalmanFilter = kalmanFilter
        self.template = template
        self.decVar = decVar
        mw.csvLoadObs([obsID])
        self.observation = modelAccel(mw.observations[obsID], template)
        self.X_ = np.asarray(self.observation[var])
        np.random.seed(seed=3)
        self.X_2 = self.X_ + np.sqrt(GLOBAL_NOISE)*np.random.randn(*self.X_.shape)
        np.random.seed(seed=2)
        self.X_ = self.X_ + np.sqrt(GLOBAL_NOISE)*np.random.randn(*self.X_.shape)
        self.dX_ = np.asarray(self.observation['d' + var])
        self.ddX_ = np.asarray(self.observation['acc' + var])

        self.trendCoefficients = np.polyfit(x=range(len(self.X_)),y=self.X_,deg=1)
        self.dtrend_X = np.arange(len(self.X_)) * self.trendCoefficients[0] + self.trendCoefficients[1]
        self.dtrend_dX = self.trendCoefficients[0] * 1/template['dt']
        self.X_ = self.X_ - self.dtrend_X
        self.dX_ = self.dX_ - self.dtrend_dX

        self.trendCoefficients2 = np.polyfit(x=range(len(self.X_2)),y=self.X_2,deg=1)
        self.dtrend_X2 = np.arange(len(self.X_2)) * self.trendCoefficients2[0] + self.trendCoefficients2[1]
        self.dtrend_dX2 = self.trendCoefficients2[0] * 1/template['dt']
        self.X_2 = self.X_2 - self.dtrend_X2
        self.dX_2 = self.dX_ - self.dtrend_dX2

    def costx(self, decVar):
        xDim = 3
        zDim = 1

        f = eval('kalman_' + self.kalmanFilter)(self.template, xDim, zDim, self.X_, decVar)
        (mu, cov, _, _) = f.batch_filter(self.X_)
        (x, P, k) = f.rts_smoother(mu, cov)
        return np.r_[(x[:,2] - self.ddX_)*1e-2, (x[:,1] - self.dX_)*1e-1, (x[:,0] - self.X_)*1e-2]

    def costy(self, decVar):
        xDim = 3
        zDim = 1

        f = eval('kalman_' + self.kalmanFilter)(self.template, xDim, zDim, self.X_, decVar)
        (mu, cov, _, _) = f.batch_filter(self.X_)
        (x, P, k) = f.rts_smoother(mu, cov)
        return np.r_[(x[:,2] - self.ddX_)*1e-2, (x[:,1] - self.dX_)*1e-1, (x[:,0] - self.X_)*1e-2]

def dataEstimates(data, dataID, template, obsID):
    thetaCorrection(data, [dataID])

    ktx = KalmanTuner('tune_x2', template, 'x', [template['v']*template['dt']/50, .0001 , 1000.0], obsID)
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    data =  dTrend(data, [dataID], 'Roach_x')
    data = kalmanFilterData(data, [dataID], 'Roach_x', 'tune_x2', ktx, ox[0])

    kty = KalmanTuner('tune_x2', template, 'y', [template['v']*template['dt']/100, .0001, 1000.0], obsID)
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-2, 1e-3])
    data =  dTrend(data, [dataID], 'Roach_y')
    data = kalmanFilterData(data, [dataID], 'Roach_y', 'tune_x2', kty, oy[0])

    kttheta = KalmanTuner('tune_x2', template, 'theta', [template['theta']*template['dt']/50, .0001, 1000.0], obsID)
    otheta = spo.leastsq(kttheta.costx, kttheta.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    data = kalmanFilterData(data, [dataID], 'Roach_theta', 'tune_x2', kttheta, otheta[0])
    return data

if __name__ == "__main__":
    #generating figures

    labelTrials = pickle.load( open( "RobertFull2/labelTrials.p", "rb" ) )
    labelTrials['test'] = 'testing'

    samples = 50
    # Load obs from LLS trial
    mw.csvLoadObs([0, 1, 2])
    mw.csvLoadData([0])
    template = mc.jsonLoadTemplate('templateControl')

    #acceleration estimates
    '''
    data = dataEstimates(mw.data, 0, template)
    '''

    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    template = mc.jsonLoadTemplate('templateControl')
    estimatesTable = generateEstimateTable('control', mw.data, dataIDs, samples, template, 0)

    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    mw.csvLoadData(dataIDs)
    #template = mc.jsonLoadTemplate('templateMass')
    estimatesTable = generateEstimateTable('mass', mw.data, dataIDs, samples, template, 1)

    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    mw.csvLoadData(dataIDs)
    #template = mc.jsonLoadTemplate('templateInertia')
    estimatesTable = generateEstimateTable('inertia', mw.data, dataIDs, samples, template, 2)

    #Saving Tarsus trajectories
    '''
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    generateFigsTarsus('control', mw.data, dataIDs, samples, template)
    '''
    #testing shai kalman
    '''
    dataIDs = mo.treatments.query("Treatment == 'control' and AnimalID == 7").index
    generateFigs('test', 'b', mw.data, dataIDs, samples, template)
    '''
    #run kalman tuner
    '''
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    generateFigs('150828-control', 'b', mw.data, dataIDs, samples, template)

    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    generateFigs('150828-mass', 'r', mw.data, dataIDs, samples, template)

    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    generateFigs('150828-inertia', 'g', mw.data, dataIDs, samples, template)


    for i in [2,4,6,7,8,9]:
        dataIDs = mo.treatments.query("Treatment == 'control' and AnimalID == " + str(i)).index
        generateFigs('250715-c-' + str(i), 'b', mw.data, dataIDs, samples, template)

    for i in [2,3,4,5,6,7,8,9]:
        dataIDs = mo.treatments.query("Treatment == 'mass' and AnimalID == " + str(i)).index
        generateFigs('250715-m-' + str(i), 'r', mw.data, dataIDs, samples, template)

    for i in [1,2,4,6,7,8,9]:
        dataIDs = mo.treatments.query("Treatment == 'inertia' and AnimalID == " + str(i)).index
        generateFigs('250715-i-' + str(i), 'g', mw.data, dataIDs, samples, template)
    '''

    #testing kalman
    '''
    mw.csvLoadObs([0])
    mw.csvLoadData([0, 1, 2])
    template = mc.jsonLoadTemplate('templateControl')

    checkKalmanFilter(mw.data, 0, template, 'Roach_x', 'x')
    checkKalmanFilter(mw.data, 0, template, 'Roach_y', 'y')
    checkKalmanFilter(mw.data, 0, template, 'Roach_theta', 'theta')
    checkKalmanFilter(mw.data, 1, template, 'Roach_x', 'x')
    checkKalmanFilter(mw.data, 1, template, 'Roach_y', 'y')
    checkKalmanFilter(mw.data, 1, template, 'Roach_theta', 'theta')
    checkKalmanFilter(mw.data, 2, template, 'Roach_x', 'x')
    checkKalmanFilter(mw.data, 2, template, 'Roach_y', 'y')
    checkKalmanFilter(mw.data, 2, template, 'Roach_theta', 'theta')
    '''

    #spoofing kalman
    '''
    mw.csvLoadObs([0])
    template = mc.jsonLoadTemplate('templateControl')
    data =     spoofKalman(template, 'Roach_x', 'x')
    '''

    #legacy

    template = mc.jsonLoadTemplate('templateInertia')
    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    mw.csvLoadData(dataIDs)

    '''
    ktx = KalmanTuner('tune_x', template, 'x', [template['v']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0])
    ox = spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    #(x1, P1, x2, P2, f) = plotModelSimulate(template, 'x', 'tune_x', ktx, ox[0])
    #plotDataKalman(mw.data, dataIDs, 'Roach_x', 'tune_x', ktx, ox[0])
    #meanStats = bootstrapDataKalman(mw.data, dataIDs, 10, 'Roach_x', 'tune_x', ktx, ox[0])

    kty = KalmanTuner('tune_x', template, 'y', [template['v']*template['dt']/100, 0.0, 0.0, .0001, 0.0 , 1000.0])
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-2, 1e-2, 1e-3, 1e-3])
    #(x1, P1, x2, P2, f) = plotModelSimulate(template, 'y', 'tune_x', kty, oy[0])
    #plotDataKalman(mw.data, dataIDs, 'Roach_y', 'tune_x', kty, oy[0])
    meanStats = bootstrapDataKalman(mw.data, dataIDs, 1, 'Roach_y', 'tune_x', kty, oy[0])

    kttheta = KalmanTuner('tune_x', template, 'theta', [template['dtheta']*template['dt']/50, 0.0, 0.0, .0001, 0.0 , 1000.0], 2)
    otheta = spo.leastsq(kttheta.costx, kttheta.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-1, 1e-1, 1e-2, 1e-2])
    #(x1, P1, x2, P2, f) = plotModelSimulate(template, 'theta', 'tune_x', kttheta, otheta[0])
    #plotDataKalman(mw.data, dataIDs, 'Roach_theta', 'tune_x', kttheta, otheta[0])
    meanStats = bootstrapDataKalman(mw.data, dataIDs, 1, 'Roach_theta', 'tune_x', kttheta, otheta[0])
    '''

    pickle.dump(labelTrials, open( "RobertFull2/labelTrials.p", "wb" ))
