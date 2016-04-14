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
from scipy import integrate
import pandas as pd
import math
from collections import deque
import copy
import random
import footlocation as fl
from shrevz import util as shutil
import kalman
import tv as tv
import pwhamil as pw

#dt = .0001
GRID_SAMPLES = 64
ORDER = 3
COST_BUFFER = 10

def chop( v ):
    return v[1:]

def Aint(d, dt=.002):
    A_ = lambda q: (np.cumsum(q) - 0.5 * (q + q[0])) * dt
    y = np.identity(np.shape(d)[0])
    A_m = np.vstack([A_(r) for r in y]).T
    A2_m = np.dot(A_m, A_m)
    A2 = lambda x : np.dot(A2_m, x)
    return A2(d)

def loadTrialEstimates(label):
    return pickle.load(open(os.path.join(os.getcwd(),os.path.join('Chartrand','Trials'),label+'.p'), 'rb'))

def dTrendTrials(trials, trialIDs, column):
    '''
        Input - Takes a dictionary of trials
            array - 3 x n: 0-position, 1-velocity, 2-acceleration
        Output- Dtrended array
    '''
    for trialID in trialIDs:
        trial = trials[trialID]
        trendCoefficients = np.polyfit(x=range(len(trial[column,:])),y=trial[column,:],deg=1)
        dtrend_X = np.arange(len(trial[column,:])) * trendCoefficients[0] + trendCoefficients[1]
        trial[column,:] -= - dtrend_X
    return trials, dtrend_X

def plotChartrand(label):
    results = loadTrialEstimates(label)
    minIndex = np.argmin(results['costs'])
    print 'Best Alpha: ' + str(results['alphas'][minIndex])
    plt.plot(results['DX_c'][:,minIndex],'b')
    plt.plot(results['X'][:,2],'g')
    plt.show()
    plt.plot(results['DDX_c'][:,minIndex],'b')
    plt.plot(results['X'][:,2],'g')
    plt.show()
    for i in range(15):
        print results['alphas'][i]
        plt.plot(results['DX_c'][:,i][100:200],'b')
        plt.plot(results['X'][:,2][100:200],'g')
        plt.show()
        plt.plot(results['DDX_c'][:,i][100:200],'b')
        plt.plot(results['X'][:,2][100:200],'g')
        plt.show()

def plotChartrand2(label, color, chartrandLabel, estimatesLabel, fil):
    estimatesTable = pw.loadEstimatesTable(estimatesLabel)
    chartrandTable = pw.loadEstimatesTable(chartrandLabel)

    (posX, _, accelX) = kalman.columnTableX('Roach_x')
    (columnX, columndX, columnddX) = columnTable('Roach_x', fil)
    plt.figure(1,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_Mean'], 'k.-', lw=3., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_01'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_99'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelX + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelX + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelX + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddx (cm/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddx.svg')
    plt.show()

    (posY, _, accelY) = kalman.columnTableX('Roach_y')
    (columnX, columndX, columnddX) = columnTable('Roach_y', fil)
    plt.figure(2,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_Mean'], 'k.-', lw=3., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_01'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_99'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelY + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelY + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelY + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddy (cm/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddy.svg')

    (posTheta, _, accelTheta) = kalman.columnTableX('Roach_theta')
    (columnX, columndX, columnddX) = columnTable('Roach_theta', fil)
    plt.figure(3,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_Mean'], 'k.-', lw=3., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_01'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_99'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelTheta + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelTheta + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelTheta + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddtheta (deg/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddtheta.svg')

    (posPitch, _, accelPitch) = kalman.columnTableX('Roach_pitch')
    (columnX, columndX, columnddX) = columnTable('Roach_pitch', fil)
    plt.figure(4,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_Mean'], 'k.-', lw=3., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_01'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_99'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelPitch + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelPitch + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelPitch + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddpitch (deg/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddpitch.svg')

    (posRoll, _, accelRoll) = kalman.columnTableX('Roach_roll')
    (columnX, columndX, columnddX) = columnTable('Roach_roll', fil)
    plt.figure(5,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_Mean'], 'k.-', lw=3., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_01'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, chartrandTable[columnddX + '_99'], 'k-', lw=1., markersize=12, label='chartrand Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelRoll + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelRoll + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(estimatesTable.index, estimatesTable[accelRoll + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddroll (deg/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddroll.svg')

def J(alpha, x, dx, ddx, dt, iter=None, scale='large'):
    if iter==None:
        iter = 10**3
    if scale == 'large':
        dx_c = tv.diff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False)#,scale='large')
        ddx_c = tv.ddiff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False)#,scale='large')
        return np.sum((dx_c[COST_BUFFER:-(COST_BUFFER+1)]-dx[COST_BUFFER:-COST_BUFFER])**2)\
        , np.sum((ddx_c[COST_BUFFER:-(COST_BUFFER+1)]-ddx[COST_BUFFER:-COST_BUFFER])**2), dx_c, ddx_c
    elif scale == 'small':
        dx_c = tv.diff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False,scale='large')
        ddx_c = tv.ddiff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False,scale='large')
        return np.sum((dx_c[COST_BUFFER:-COST_BUFFER]-dx[COST_BUFFER:-COST_BUFFER])**2)\
        , np.sum((ddx_c[COST_BUFFER:-COST_BUFFER]-ddx[COST_BUFFER:-COST_BUFFER])**2), dx_c, ddx_c

def JVar(alpha, x, dx, ddx, dt, iter=None, scale='large'):
    if iter==None:
        iter = 10**3
    dx_f = np.hstack([0.,np.diff(x)])/dt
    dx_m = dx_f.mean()
    dx_f -= dx_m
    #b,a = signal.butter(2,.1)
    b,a = signal.butter(2,.3)
    ddx_f = np.concatenate( ([0], [0], np.diff( signal.filtfilt(b,a, np.diff( signal.filtfilt(b,a, x ) ) ) ) ) )/dt/dt

    if scale == 'large':
        dx_c = tv.diff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False)#,scale='large')
        ddx_c = tv.ddiff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False)#,scale='large')
        return np.abs(np.var(dx_c[COST_BUFFER:-(COST_BUFFER+1)]) - np.var(dx_f[COST_BUFFER:-COST_BUFFER])) \
        , np.abs(np.var(ddx_c[COST_BUFFER:-(COST_BUFFER+1)]) - np.var(ddx_f[COST_BUFFER:-COST_BUFFER])), dx_c, ddx_c
    elif scale == 'small':
        dx_c = tv.diff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False,scale='large')
        ddx_c = tv.ddiff(x,iter,alpha,dx=dt,plotflag=False,diagflag=False,scale='large')
        return np.abs(np.var(dx_c[COST_BUFFER:-COST_BUFFER]) - np.var(dx_f[COST_BUFFER:-COST_BUFFER])) \
        , np.abs(np.var(ddx_c[COST_BUFFER:-COST_BUFFER]) - np.var(ddx_f[COST_BUFFER:-COST_BUFFER])), dx_c, ddx_c

def dataToEstimatesTable(column, meanStats, estimatesTable, fil):
    (columnX, columndX, columnddX) = columnTable(column, fil)

    estimatesTable[columndX + '_Mean'] = meanStats[columndX + '_Mean']
    estimatesTable[columndX + '_01'] = meanStats[columndX + '_01']
    estimatesTable[columndX + '_99'] = meanStats[columndX + '_99']

    estimatesTable[columnddX + '_Mean'] = meanStats[columnddX + '_Mean']
    estimatesTable[columnddX + '_01'] = meanStats[columnddX + '_01']
    estimatesTable[columnddX + '_99'] = meanStats[columnddX + '_99']
    return estimatesTable

def bootstrap(data, dataIDs, samples, column, fil):
    random.seed(2)
    means = {}
    (columnX, columndX, columnddX) = columnTable(column, fil)
    for i in range(samples):
        randomDataIDs = np.random.choice(dataIDs, len(dataIDs))
        mean = pd.DataFrame(columns=[columndX, columnddX], index=range(GRID_SAMPLES))
        mean[columndX] = 0.0
        mean[columnddX] = 0.0
        phase = np.array([[]])
        valuesdX = np.array([[]])
        valuesddX = np.array([[]])
        for dataID in randomDataIDs:
            phase = np.hstack((phase, data[dataID]['Roach_xv_phase'][0:283].dropna()[np.newaxis,:]))
            valuesdX = np.hstack((valuesdX, data[dataID][columndX][0:283].dropna()[np.newaxis,:]))
            valuesddX = np.hstack((valuesddX, data[dataID][columnddX][0:283].dropna()[np.newaxis,:]))
        fsdX = shutil.FourierSeries()
        fsddX = shutil.FourierSeries()
        fsdX.fit(ORDER, copy.deepcopy(phase), valuesdX)
        fsddX.fit(ORDER, copy.deepcopy(phase), valuesddX)
        grid  = np.arange(GRID_SAMPLES)[np.newaxis,:] * (2*np.pi)/GRID_SAMPLES
        mean[columndX] = fsdX.val(grid)[:,0].real
        mean[columnddX] = fsddX.val(grid)[:,0].real
        #mean = mean/len(dataIDs)
        means[i] = mean
    return means

def phaseAverage(bins, phase, values):
    phase = phase % (2 * np.pi)
    #obs-observations, d-dimensions
    obs, d = np.shape(values)
    phaseBins = np.linspace(0, 2*np.pi, num=bins+1)
    inds = np.digitize(phase[:,0], phaseBins) - 1
    phaseAverages = np.zeros((bins, d))
    counts = np.zeros(bins)
    for i in range(obs):
        #print 'phase: ' + str(phase[i,0])
        index = inds[i]
        #print 'index: ' + str(index)
        counts[index] += 1
        phaseAverages[index,:] += values[i,:]
    for i in range(bins):
        phaseAverages[i,:] = phaseAverages[i,:]/counts[i]
    return phaseAverages

def bootstrapPhaseAverage(data, dataIDs, samples, column, fil):
    random.seed(2)
    means = {}
    (columnX, columndX, columnddX) = columnTable(column, fil)
    for i in range(samples):
        randomDataIDs = np.random.choice(dataIDs, len(dataIDs))
        mean = pd.DataFrame(columns=[columndX, columnddX], index=range(GRID_SAMPLES))
        mean[columndX] = 0.0
        mean[columnddX] = 0.0
        phase = deque()
        valuesdX = deque()
        valuesddX = deque()
        for dataID in randomDataIDs:
            phase.append(data[dataID]['Roach_xv_phase'][0:283].dropna()[:,np.newaxis])
            valuesdX.append(data[dataID][columndX][0:283].dropna()[:,np.newaxis])
            valuesddX.append(data[dataID][columnddX][0:283].dropna()[:,np.newaxis])
        phaseArray = np.vstack(phase)
        valuesdXArray = np.vstack(valuesdX)
        valuesddXArray = np.vstack(valuesddX)
        phaseAverages = phaseAverage(64, phaseArray, np.hstack((valuesdXArray, valuesddXArray)))
        mean[columndX] = phaseAverages[:,0]
        mean[columnddX] = phaseAverages[:,1]
        #mean = mean/len(dataIDs)
        means[i] = mean
    return means

def meanStats(means, column, fil):
    meansList_dx = {}
    meansList_ddx = {}
    (columnX, columndX, columnddX) = columnTable(column, fil)
    for i in range(GRID_SAMPLES):
        meansList_dx[i] = deque()
        meansList_ddx[i] = deque()

    for meanID in means.keys():
        for i in range(GRID_SAMPLES):
            meansList_dx[i].append(means[meanID].ix[i, columndX])
            meansList_ddx[i].append(means[meanID].ix[i, columnddX])

    columns=[columndX + '_Mean', columndX + '_01', columndX + '_99' \
    ,columnddX + '_Mean', columnddX + '_01', columnddX + '_99']
    meanStats = pd.DataFrame(columns=columns, index=range(GRID_SAMPLES))

    for i in range(GRID_SAMPLES):
        dxList = pd.Series(meansList_dx[i])
        ddxList = pd.Series(meansList_ddx[i])
        if all(np.isnan(a) for a in dxList):
            continue
        meanStats.ix[i, columndX + '_Mean'] = dxList.mean()
        meanStats.ix[i, columnddX + '_Mean'] = ddxList.mean()
        meanStats.ix[i, columndX + '_99'] = dxList.quantile(q=[.01]).ix[.01]
        meanStats.ix[i, columnddX + '_01'] = ddxList.quantile(q=[.01]).ix[.01]
        meanStats.ix[i, columndX + '_01'] = dxList.quantile(q=[.99]).ix[.99]
        meanStats.ix[i, columnddX + '_99'] = ddxList.quantile(q=[.99]).ix[.99]

    return meanStats

def columnTableX(column):
    columnX = column + '_ChartrandX'
    columndX = column + '_ChartrandDX'
    columnddX = column + '_ChartrandDDX'
    return (columnX, columndX, columnddX)

def columnTable(column, method, suffix=None):
    if method == 'Butter':
        columnX = column + '_ButterX'
        columndX = column + '_ButterDX'
        columnddX = column + '_ButterDDX'
    elif method == 'Chartrand':
        columnX = column + '_ChartrandX'
        columndX = column + '_ChartrandDX'
        columnddX = column + '_ChartrandDDX'
    elif method == 'Kalman':
        columnX = column + '_KalmanX'
        columndX = column + '_KalmanDX'
        columnddX = column + '_KalmanDDX'
    if suffix != None:
        columnX += suffix
        columndX += suffix
        columnddX += suffix
    return (columnX, columndX, columnddX)

def chatrandFilterData(data, dataIDs, column, alpha, iterations, label):
    dt = .002
    (columnX, columndX, columnddX) = columnTableX(column)
    filteredData = {}
    for dataID in dataIDs:
        currentData = data[dataID]
        indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(currentData, 284, 'pre')
        X_ = np.asarray(currentData[column].dropna())
        DX = tv.diff(X_,iterations,alpha[0],dx=dt,plotflag=False,diagflag=False,scale='small')
        DDX = tv.ddiff(X_,iterations,alpha[1],dx=dt,plotflag=False,diagflag=False,scale='small')
        currentData[columndX] = float('nan')
        currentData[columnddX] = float('nan')
        diff = indexEnd - indexStart
        currentData[columndX][indexStart:indexEnd] = DX[:diff]
        currentData[columnddX][indexStart:indexEnd] = DDX[:diff]
        filteredData[dataID] = np.vstack((X_[:diff], DX[:diff], DDX[:diff]))
    pickle.dump(filteredData, open(os.path.join(os.getcwd(), os.path.join('Chartrand','Trials'),label+'-'+column+'.p'), 'wb'))
    return data

def butterFilter(x, d, c, b, a):
    dt = .002
    dx = np.hstack([0.,np.diff(x)])/dt
    dx_m = dx.mean()
    dx -= dx_m
    b,a = signal.butter(b,a)
    dx_f = signal.filtfilt(b,a, dx)
    d,c = signal.butter(d,c)
    ddx_f = np.concatenate( ([0], [0], np.diff( signal.filtfilt(d,c, np.diff( signal.filtfilt(d,c, x ) ) ) ) ) )/dt/dt
    return x, dx_f, ddx_f

def butterFilterData(data, dataIDs, column, label):
    dt = .002
    (columnX, columndX, columnddX) = columnTable(column, 'Butter')
    filteredData = {}
    for dataID in dataIDs:
        currentData = data[dataID]
        indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(currentData, 284, 'pre')
        X_ = np.asarray(currentData[column][0:284].dropna())
        if column == 'Roach_x':
            X, DX, DDX, = butterFilter(X_, 3, 0.15, 2, 0.15, 2, 0.15)
        elif column == 'Roach_y':
            X, DX, DDX, = butterFilter(X_, 3, 0.1, 4, 0.15, 2, 0.15)
        elif column == 'Roach_theta':
            X, DX, DDX, = butterFilter(X_, 3, 0.1, 4, 0.1, 2, 0.15)
        else:
            X, DX, DDX, = butterFilter(X_, 10, 0.1, 4, 0.1, 2, 0.15)

        currentData[columnX] = float('nan')
        currentData[columndX] = float('nan')
        currentData[columnddX] = float('nan')
        currentData[columnX][indexStart:indexEnd] = X
        currentData[columndX][indexStart:indexEnd] = DX
        currentData[columnddX][indexStart:indexEnd] = DDX
        filteredData[dataID] = np.vstack((X, DX, DDX))

    pickle.dump(filteredData, open(os.path.join(os.getcwd(), os.path.join('Chartrand','Trials'),label+'-'+column+'.p'), 'wb'))
    return data

def bootstrapData(data, dataIDs, samples, column, alpha, label, fil, method='fs', iterations=None):
    if fil=='Chartrand':
        data = chatrandFilterData(data, dataIDs, column, alpha, iterations, label)
    elif fil=='Butter':
        data = butterFilterData(data, dataIDs, column, label)
    if method=='fs':
        means = bootstrap(data, dataIDs, samples, column, fil)
    elif method=='pa':
        means = bootstrapPhaseAverage(data, dataIDs, samples, column, fil)
    meansR = copy.deepcopy(means)
    mStats = meanStats(means, column, fil)
    return mStats, meansR

def generateEstimateTable(label, data, dataIDs, samples, template, obsID, fil, method, iterations=None, alphas=None):
    data, thetaMean = kalman.thetaCorrection(data, dataIDs)
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'Roach_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'Roach_y')
    '''
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody1_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody2_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody3_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody4_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody5_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'TarsusBody6_x')
    '''
    for dataID in dataIDs:
        data[dataID]['Roach_pitch'] =  data[dataID]['Roach_pitch'] * (2 * np.pi)/360
        data[dataID]['Roach_roll'] =  data[dataID]['Roach_roll'] * (2 * np.pi)/360

    meansDict = {}

    if alphas==None:
        alphas = [1e-8, 1e-8]

    meanStatsX, meansDict['Roach_x'] = bootstrapData(data, dataIDs, samples, 'Roach_x', alphas, label, fil, method, iterations=iterations)
    meanStatsY, meansDict['Roach_y'] = bootstrapData(data, dataIDs, samples, 'Roach_y', alphas, label, fil, method, iterations=iterations)
    meanStatsTheta, meansDict['Roach_theta'] = bootstrapData(data, dataIDs, samples, 'Roach_theta', alphas, label, fil, method, iterations=iterations)
    meanStatsPitch, meansDict['Roach_pitch'] = bootstrapData(data, dataIDs, samples, 'Roach_pitch', alphas, label, fil, method, iterations=iterations)
    meanStatsRoll, meansDict['Roach_roll'] = bootstrapData(data, dataIDs, samples, 'Roach_roll', alphas, label, fil, method, iterations=iterations)
    meanStatsTB1_x, meansDict['TarsusBody1_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody1_x', alphas, label, fil, method, iterations=iterations)
    meanStatsTB2_x, meansDict['TarsusBody2_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody2_x', alphas, label, fil, method, iterations=iterations)
    meanStatsTB3_x, meansDict['TarsusBody3_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody3_x', alphas, label, fil, method, iterations=iterations)
    meanStatsTB4_x, meansDict['TarsusBody4_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody4_x', alphas, label, fil, method, iterations=iterations)
    meanStatsTB5_x, meansDict['TarsusBody5_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody5_x', alphas, label, fil, method, iterations=iterations)
    meanStatsTB6_x, meansDict['TarsusBody6_x'] = bootstrapData(data, dataIDs, samples, 'TarsusBody6_x', alphas, label, fil, method, iterations=iterations)

    estimatesTable = pd.DataFrame(index=range(64))

    estimatesTable = dataToEstimatesTable('Roach_x', meanStatsX, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('Roach_y', meanStatsY, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('Roach_theta', meanStatsTheta, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('Roach_pitch', meanStatsPitch, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('Roach_roll', meanStatsRoll, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody1_x', meanStatsTB1_x, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody2_x', meanStatsTB2_x, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody3_x', meanStatsTB3_x, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody4_x', meanStatsTB4_x, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody5_x', meanStatsTB5_x, estimatesTable, fil)
    estimatesTable = dataToEstimatesTable('TarsusBody6_x', meanStatsTB6_x, estimatesTable, fil)

    #kalman.fourierSeriesToFile(meansDict, varList, samples, label)
    pickle.dump(estimatesTable, open(os.path.join(os.getcwd(), 'Estimates',label+'.p'), 'wb'))
    return estimatesTable

def gridSearch(f, x, dx, ddx, label='test'):
    SCALE = 15
    alpha_o = 0.
    minima = float('inf')
    dim, = np.shape(x)
    X = np.vstack((x, dx, ddx)).T
    alphas = np.zeros(SCALE)
    costs = np.zeros(SCALE)
    results = {}
    DX_c = np.zeros((dim, SCALE))
    DDX_c = np.zeros((dim, SCALE))
    for i in range(SCALE):
        alpha = 1000.*10.**-i
        c_dx, c_ddx, dx_c, ddx_c = f(alpha)
        DX_c[:,i] = dx_c
        DDX_c[:,i] = ddx_c
        currentCost = 1.e-1 * c_dx + c_ddx
        alphas[i] = alpha
        costs[i] = currentCost
        if currentCost < minima:
            minima = currentCost
            alpha_o = alpha
    results['alphas'] = alphas
    results['costs'] = costs
    results['DDX_c'] = DDX_c
    results['DX_c'] = DX_c
    results['X'] = X
    pickle.dump(results, open(os.path.join(os.getcwd(), 'Chartrand',label+'.p'), 'wb'))
    return alpha_o

def optimizeChartrand(J, x, dx, ddx, dt, iterations=100, alpha0=3.5e-3, scale='large', label='test'):
    def f(alpha):
        print 'alpha: ' + str(alpha)
        c_dx, c_ddx, dx_c, ddx_c = JVar(alpha, x, dx, ddx, dt, iter=iterations)
        print 'cost: ' + str(1.e-1 * c_dx + c_ddx)
        #cost = JVar(alpha, x, dx, ddx, dt, iter=iterations)
        #print 'cost: ' + str(cost)
        return c_dx, c_ddx, dx_c[:-1], ddx_c[:-1]
        #return cost

    #print spo.bisect(f, -0.5+1e-12, 0.5-1e-12, maxiter=20)
    #funct = lambda x : f(x[0])
    #results = spo.minimize_scalar(f, bounds=(0., 1.), options={'maxiter':50}, method='bounded', tol=1e-5)
    #alpha_o = results['x']
    alpha_o = gridSearch(f, x, dx, ddx, label=label)
    print 'Alpha Optimal: ' + str(alpha_o)
    '''
    c_dx, c_ddx, dx_c, ddx_c = JVar(alpha_o, x, dx, ddx, dt, iter=iterations, scale=scale)
    plt.plot(dx_c, 'b')
    plt.plot(dx, 'g')
    plt.show()
    plt.plot(ddx_c, 'b')
    plt.plot(ddx, 'g')
    plt.show()
    '''

def tuneChartrand(observation, template, iterations):
    kalman.modelAccel(observation, template)
    observation['ddx'] = observation['accx']
    observation['ddy'] = observation['accy']
    observation['ddtheta'] = observation['acctheta']

    dt = .0001
    obs = mw.observations[0]
    x = np.array(obs['x'][100:1000]-np.mean(obs['x'][100:1000])) + 1e-4*np.random.randn(900)
    dx = np.array(obs['dx'][100:1000]-np.mean(obs['dx'][100:1000]))
    ddx = np.array(obs['ddx'][100:1000])
    optimizeChartrand(J, x, dx, ddx, dt, iterations=iterations, alpha0=3.5e-3, scale='large', label='mse-100-lores-noise')

def testChartrand(observation, template, iterations):
    kalman.modelAccel(observation, template)
    observation['ddx'] = observation['accx']
    observation['ddy'] = observation['accy']
    observation['ddtheta'] = observation['acctheta']

    dt = .002
    #print JVar(3.5e-3, x, dx, ddx, dt, iter=2)

    #No Noise Large
    #optimal var: 1e-07
    #filter var: 0.00879236610709
    #optimal mse: 1.0

    #No Noise Small
    #optimal var:
    #filter var:
    #optimal mse: 0.499038200269

    #Noise Small
    #optimal var:
    #filter var:
    #optimal mse: 0.499038200269

    alpha = 1e-8
    #alpha = 1e-11
    obs = mw.observations[0]
    x = np.array(obs['x'][100:1000]-np.mean(obs['x'][100:1000])) + 1e-4*np.random.randn(900)
    dx = np.array(obs['dx'][100:1000])#-np.mean(obs['dx'][100:1000]))
    ddx = np.array(obs['ddx'][100:1000])
    c_dx, c_ddx, dx_c, ddx_c = J(alpha, x, dx, ddx, dt, iter=iterations, scale='small')

    #ktx = kalman.KalmanTuner('tune_x2', template, 'x', [template['v']*template['dt']/50, .0001 , 1000.0], 0)
    #ox = kalman.spo.leastsq(ktx.costx, ktx.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-1, 1e-2])
    #(mu, cov, x_k, P_k) = kalman.dataKalman(x, 'tune_x2', ox[0])

    alpha = 1e-6
    #alpha = 1e-11
    obs = mw.observations[0]
    y = np.array(obs['y'][100:1000]-np.mean(obs['y'][100:1000])) + 1e-4*np.random.randn(900)
    dy = np.array(obs['dy'][100:1000]-np.mean(obs['dy'][100:1000]))
    ddy = np.array(obs['ddy'][100:1000])
    c_dy, c_ddy, dy_c, ddy_c = J(alpha, y, dy, ddy, dt, iter=iterations, scale='small')

    #kty = kalman.KalmanTuner('tune_x2', template, 'y', [template['v']*template['dt']/100, .0001, 1000.0], 0)
    #oy = kalman.spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e-2, 1e-3])
    #(mu, cov, y_k, P_k) = kalman.dataKalman(y, 'tune_x2', oy[0])

    alpha = 1e-6
    #alpha = 1e-11
    obs = mw.observations[0]
    theta = np.array(obs['theta'][100:1000]-np.mean(obs['theta'][100:1000])) + 1e-4*np.random.randn(900)
    dtheta = np.array(obs['dtheta'][100:1000]-np.mean(obs['dtheta'][100:1000]))
    ddtheta = np.array(obs['ddtheta'][100:1000])
    c_dtheta, c_ddtheta, dtheta_c, ddtheta_c = J(alpha, theta, dtheta, ddtheta, dt, iter=iterations, scale='small')

    mw.csvLoadObs([0])
    obs = mw.observations[0]
    dx = np.array(obs['dx'][100:1000])
    dy = np.array(obs['dy'][100:1000])
    dd = np.array(obs['dtheta'][100:1000])

    plt.figure(1)
    plt.plot(y[100:200], 'b.-')
    plt.plot(np.array(obs['y'][100:200]),'go')
    #plt.plot(x_k[100:200,2])
    plt.title('$y$', fontsize=18)
    plt.show()

    plt.figure(1)
    plt.plot(theta[100:200], 'b.-')
    plt.plot(np.array(obs['theta'][100:200]),'g')
    #plt.plot(x_k[100:200,2])
    plt.title('$theta$', fontsize=18)
    plt.show()

    plt.figure(1)
    plt.plot(dx[100:200], 'b.-')
    plt.plot(dx_c[100:200], 'g.')
    #plt.plot(x_k[100:200,2])
    plt.title('$\dot{x}$', fontsize=18)
    plt.show()

    plt.figure(2)
    plt.plot(ddx[100:200], 'b.-')
    plt.plot(ddx_c[100:200], 'g.')
    plt.title('$\ddot{x}$', fontsize=18)
    #plt.plot(x_k[100:200,2])
    plt.show()

    plt.figure(3)
    plt.plot(dx, 'b.-')
    plt.plot(dx_c, 'g.')
    plt.title('$\dot{x}$', fontsize=18)
    #plt.plot(x_k[100:200,2])
    plt.show()

    plt.figure(4)
    plt.plot(ddx, 'b.-')
    plt.plot(ddx_c, 'g.')
    plt.title('$\ddot{x}$', fontsize=18)
    #plt.plot(x_k[100:200,2])
    plt.show()

    plt.figure(5)
    plt.plot(dy[100:200], 'b.-')
    plt.plot(dy_c[100:200], 'g.')
    plt.title('$\dot{y}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(6)
    plt.plot(ddy[100:200], 'b.-')
    plt.plot(ddy_c[100:200], 'g.')
    plt.title('$\ddot{y}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(7)
    plt.plot(dy, 'b.-')
    plt.plot(dy_c, 'g.')
    plt.title('$\dot{y}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(8)
    plt.plot(ddy, 'b.-')
    plt.plot(ddy_c, 'g.')
    plt.title('$\ddot{y}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(9)
    plt.plot(dtheta[100:200], 'b.-')
    plt.plot(dtheta_c[100:200], 'g.')
    plt.title('$\dot{theta}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(10)
    plt.plot(ddtheta[100:200], 'b.-')
    plt.plot(ddtheta_c[100:200], 'g.')
    plt.title('$\ddot{theta}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(11)
    plt.plot(dtheta, 'b.-')
    plt.plot(dtheta_c, 'g.')
    plt.title('$\dot{theta}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

    plt.figure(12)
    plt.plot(ddtheta, 'b.-')
    plt.plot(ddtheta_c, 'g.')
    plt.title('$\ddot{theta}$', fontsize=18)
    #plt.plot(y_k[100:200,2])
    plt.show()

def histogramPhaseBins(data, dataIDs):
    counts = deque()
    for dataID in dataIDs:
        currentData = data[dataID]
        phase = np.array(currentData['Roach_xv_phase'][0:284].dropna())
        phase = phase % (2 * np.pi)
        phaseBins = np.linspace(0, 2*np.pi, num=GRID_SAMPLES+1)
        inds = np.digitize(phase, phaseBins) - 1
        counts.append(inds)

    plt.xlim(0,64)
    plt.hist(np.hstack(counts), bins=64, facecolor='green', alpha=0.75)
    plt.tight_layout()
    plt.show()

def residuals(data, dataIDs, c, b, k, column):
    rK = deque()
    rC = deque()
    rB = deque()
    for dataID in dataIDs:
        cData = data[dataID]
        xK = Aint(k[dataID][2,:])
        xC = Aint(c[dataID][2,:])
        xB = Aint(b[dataID][2,:])
        x = cData[column][0:284].dropna()
        #rK.append(x - xK)
        #rC.append(x - xC)
        #rB.append(x - xB)
        rK.append(x - np.array(x)[0] - xK)
        rC.append(x - np.array(x)[0] - xC)
        rB.append(x - np.array(x)[0] - xB)
    return np.concatenate(rC), np.concatenate(rB), np.concatenate(rK)

def histogramResiduals(data, dataIDs, kalmanLabel, chartrandLabel, butterLabel):
    column = 'Roach_x'
    kalmanTrialsX = loadTrialEstimates(kalmanLabel + '-' + column)
    chartrandTrialsX = loadTrialEstimates(chartrandLabel + '-' + column)
    butterTrialsX = loadTrialEstimates(butterLabel + '-' + column)
    xC, xB, xK = residuals(data, dataIDs, chartrandTrialsX, butterTrialsX, kalmanTrialsX, 'Roach_x')
    bins = np.linspace(-20, 20, 100)
    plt.hist(xC, bins, normed=1, facecolor='green', alpha=0.5, label='Chartrand')
    plt.hist(xB, bins, normed=1, facecolor='blue', alpha=0.5, label='Butter')
    plt.hist(xK, bins, normed=1, facecolor='red', alpha=0.5, label='Kalman')
    plt.legend(loc='upper right')
    plt.show()

    column = 'Roach_y'
    kalmanTrialsX = loadTrialEstimates(kalmanLabel + '-' + column)
    chartrandTrialsX = loadTrialEstimates(chartrandLabel + '-' + column)
    butterTrialsX = loadTrialEstimates(butterLabel + '-' + column)
    xC, xB, xK = residuals(data, dataIDs, chartrandTrialsX, butterTrialsX, kalmanTrialsX, 'Roach_x')
    bins = np.linspace(-20, 20, 100)
    plt.hist(xC, bins, normed=1, facecolor='green', alpha=0.5, label='Chartrand')
    plt.hist(xB, bins, normed=1, facecolor='blue', alpha=0.5, label='Butter')
    plt.hist(xK, bins, normed=1, facecolor='red', alpha=0.5, label='Kalman')
    #plt.legend(loc='upper right')
    plt.show()

    column = 'Roach_theta'
    kalmanTrialsX = loadTrialEstimates(kalmanLabel + '-' + column)
    chartrandTrialsX = loadTrialEstimates(chartrandLabel + '-' + column)
    butterTrialsX = loadTrialEstimates(butterLabel + '-' + column)
    xC, xB, xK = residuals(data, dataIDs, chartrandTrialsX, butterTrialsX, kalmanTrialsX, 'Roach_x')
    bins = np.linspace(-20, 20, 100)
    plt.hist(xC, bins, normed=1, facecolor='green', alpha=0.5, label='Chartrand')
    plt.hist(xB, bins, normed=1, facecolor='blue', alpha=0.5, label='Butter')
    plt.hist(xK, bins, normed=1, facecolor='red', alpha=0.5, label='Kalman')
    #plt.legend(loc='upper right')
    plt.show()

def violinResiduals(data, dataIDs, kalmanLabel, chartrandLabel, butterLabel, column):
    kalmanTrialsX = loadTrialEstimates(kalmanLabel + '-' + column)
    butterTrialsX = loadTrialEstimates(butterLabel + '-' + column)

    #kalmanTrialsX, _ = dTrendTrials(kalmanTrialsX, kalmanTrialsX.keys(), 2)
    #butterTrialsX, _ = dTrendTrials(butterTrialsX, butterTrialsX.keys(), 2)
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'Roach_x')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'Roach_y')
    data, dtrend_X =  kalman.dTrend(data, dataIDs, 'Roach_theta')

    trials = []
    for alpha in range(10):
        cL = chartrandLabel + '-1e-' + str(alpha)
        chartrandTrialsX = loadTrialEstimates(cL + '-' + column)
        #chartrandTrialsX, _ = dTrendTrials(chartrandTrialsX, chartrandTrialsX.keys(), 2)
        xC, xB, xK = residuals(data, dataIDs, chartrandTrialsX, butterTrialsX, kalmanTrialsX, 'Roach_x')
        trials.append(xC)
    trials = trials + [xK, xB]

    ax = plt.gca()
    pos = range(len(trials))
    ax.violinplot(trials, pos, points=1000, vert=True, widths=1.0,
                      showmeans=True, showextrema=True,
                      bw_method=0.5)
    plt.show()

def costChartrand(dataColumn, estimatesColumn, chartrandColumn):
    return np.var(dataColumn - estimatesColumn) - np.var(dataColumn - Aint(chartrandColumn))

def varianceCostHelper(kalmanColumn, chartrandColumn, filterColumn):
    return np.abs(np.var(kalmanColumn - filterColumn) - np.var(chartrandColumn - filterColumn))

def varianceCost(estimatesLabel, chartrandLabel):
    estimatesTrialsX = loadTrialEstimates(estimatesLabel + '-Roach_x')
    chartrandTrialsX = loadTrialEstimates(chartrandLabel + '-Roach_x')
    cX = np.hstack(chartrandTrialsX.values())
    eX = np.hstack(estimatesTrialsX.values())
    costX = costChartrand(cX[0,:], eX[2,:], cX[2,:])

    estimatesTrialsY = loadTrialEstimates(estimatesLabel + '-Roach_y')
    chartrandTrialsY = loadTrialEstimates(chartrandLabel + '-Roach_y')
    cY = np.hstack(chartrandTrialsY.values())
    eY = np.hstack(estimatesTrialsY.values())
    costY = costChartrand(cY[0,:], eY[2,:], cY[2,:])

    estimatesTrialsTheta = loadTrialEstimates(estimatesLabel + '-Roach_theta')
    chartrandTrialsTheta = loadTrialEstimates(chartrandLabel + '-Roach_theta')
    cTheta = np.hstack(chartrandTrialsTheta.values())
    eTheta = np.hstack(estimatesTrialsTheta.values())
    costTheta = costChartrand(cTheta[0,:], eTheta[2,:], cTheta[2,:])

    '''
    kalmanTable = pw.loadEstimatesTable(kalmanLabel)
    chartrandTable = pw.loadEstimatesTable(chartrandLabel)
    filterTable = pw.loadEstimatesTable(filterLabel)

    (columnX, columndXButter, columnddXButter) = columnTable('Roach_x', 'Butter', '_Mean')
    (columnX, columndXKalman, columnddXKalman) = columnTable('Roach_x', 'Kalman', '_Mean')
    (columnX, columndXChartrand, columnddXChartrand) = columnTable('Roach_x', 'Chartrand', '_Mean')
    costX = varianceCostHelper(kalmanTable[columnddXKalman], chartrandTable[columnddXChartrand], filterTable[columnddXButter])

    (columnX, columndXButter, columnddXButter) = columnTable('Roach_y', 'Butter', '_Mean')
    (columnX, columndXKalman, columnddXKalman) = columnTable('Roach_y', 'Kalman', '_Mean')
    (columnX, columndXChartrand, columnddXChartrand) = columnTable('Roach_y', 'Chartrand', '_Mean')
    costY = varianceCostHelper(kalmanTable[columnddXKalman], chartrandTable[columnddXChartrand], filterTable[columnddXButter])

    (columnX, columndXButter, columnddXButter) = columnTable('Roach_theta', 'Butter', '_Mean')
    (columnX, columndXKalman, columnddXKalman) = columnTable('Roach_theta', 'Kalman', '_Mean')
    (columnX, columndXChartrand, columnddXChartrand) = columnTable('Roach_theta', 'Chartrand', '_Mean')
    costTheta = varianceCostHelper(kalmanTable[columnddXKalman], chartrandTable[columnddXChartrand], filterTable[columnddXButter])
    '''
    return costX, costY, costTheta

def butterFilter(x, f, e, d, c, b, a):
    dt = .002
    b,a = signal.butter(b,a)
    x_f = signal.filtfilt(b,a, x)
    dx = np.hstack([0.,np.diff(x)])/dt
    dx_m = dx.mean()
    dx -= dx_m
    d,c = signal.butter(d,c)
    dx_f = signal.filtfilt(d,c, dx)
    #b,a = signal.butter(2,.1)
    f,e = signal.butter(f,e)
    ddx_f = np.concatenate( ([0], [0], np.diff( signal.filtfilt(d,c, np.diff( signal.filtfilt(f,e, x ) ) ) ) ) )/dt/dt
    return np.vstack((x_f, dx_f, ddx_f))

def tuneButter(observation):
    kalman.modelAccel(observation, template)
    observation['ddx'] = observation['accx']
    observation['ddy'] = observation['accy']
    observation['ddtheta'] = observation['acctheta']

    l = len(observation.index)
    X = butterFilter(observation['x'] + 1e-3*np.random.randn(l), 10, 0.15, 2, 0.15, 2, 0.15)
    Y = butterFilter(observation['y'] + 1e-3*np.random.randn(l), 10, 0.15, 4, 0.15, 2, 0.15)
    Theta = butterFilter(observation['theta'] + 1e-3*np.random.randn(l), 10, 0.1, 4, 0.1, 2, 0.15)

    plt.plot(X[1,:][100:200], 'g-')
    plt.plot(observation['dx'][100:200] - np.mean(observation['dx']), 'b-')
    plt.show()
    plt.plot(X[2,:][100:200], 'g')
    plt.plot(observation['ddx'][100:200], 'b')
    plt.show()
    plt.plot(Y[1,:][100:200], 'g')
    plt.plot(observation['dy'][100:200] - np.mean(observation['dy']), 'b')
    plt.show()
    plt.plot(Y[2,:][100:200], 'g')
    plt.plot(observation['ddy'][100:200], 'b')
    plt.show()
    plt.plot(Theta[1,:][100:200], 'g')
    plt.plot(observation['dtheta'][100:200] - np.mean(observation['dtheta']), 'b')
    plt.show()
    plt.plot(Theta[2,:][100:200], 'g')
    plt.plot(observation['ddtheta'][100:200], 'b')
    plt.show()

def plotAlpha(filt, method, treatment, color, label):
    alphaVector = [4,5,6,7,8]
    costX = []
    costY = []
    costTheta = []
    for alpha in alphaVector:
        '''
        cX, cY, cTheta = varianceCost('kalman-' + method + '-' + treatment, \
        'chartrand-' + method + '-' + treatment +'-' + '1e-' + str(alpha), \
        'butter-' + method + '-' + treatment)
        '''
        cX, cY, cTheta = varianceCost(filt + '-' + method + '-' + treatment, \
        'chartrand-' + method + '-' + treatment +'-' + '1e-' + str(alpha))
        costX.append(cX)
        costY.append(cY)
        costTheta.append(cTheta)

    plt.plot(alphaVector, costX, 'o-', color=color)
    plt.xlim(3.5,8.5)
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddx.svg')
    plt.show()

    plt.plot(alphaVector, costY, 'o-', color=color)
    plt.xlim(3.5,8.5)
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddy.svg')
    plt.show()

    plt.plot(alphaVector, costTheta, 'o-', color=color)
    plt.xlim(3.5,8.5)
    plt.tight_layout()
    plt.savefig('Chartrand/Figures/' + label + '-ddtheta.svg')
    plt.show()

if __name__ == '__main__':
    saveDir = 'StableOrbit'

    varList = ['x','y','theta']

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)

    mw.csvLoadObs([0,1,2])
    template = mc.jsonLoadTemplate('templateControl')
    samples = 50
    iterations = 1000

    '''
    plotAlpha('kalman', 'fs', 'control', 'b', '160304-fs-alpha-control')
    plotAlpha('kalman', 'fs', 'mass', 'r', '160304-fs-alpha-mass')
    plotAlpha('kalman', 'fs', 'inertia', 'g', '160304-fs-alpha-inertia')
    '''

    '''
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    template = mc.jsonLoadTemplate('templateControl')
    generateEstimateTable('butter-fs-control', mw.data, dataIDs, samples, iterations, template, 'Butter', iterations=iterations, method='fs')

    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    template = mc.jsonLoadTemplate('templateMass')
    generateEstimateTable('butter-fs-mass', mw.data, dataIDs, samples, iterations, template, 'Butter', iterations=iterations, method='fs')

    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    template = mc.jsonLoadTemplate('templateInertia')
    generateEstimateTable('butter-fs-inertia', mw.data, dataIDs, samples, iterations, template, 'Butter', iterations=iterations, method='fs')
    '''

    #tuneButter(mw.observations[0])
    #testChartrand(mw.observations[0], template, iterations)
    #plotChartrand('mse-100-lores-nonoise')

    #violinplot of resdiuals for chartand 1e-(0:9) alpha, kalman, butter
    '''
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    violinResiduals(mw.data, dataIDs, 'kalman-fs-control', 'chartrand-fs-control', 'butter-fs-control', 'Roach_x')
    violinResiduals(mw.data, dataIDs, 'kalman-fs-control', 'chartrand-fs-control', 'butter-fs-control', 'Roach_y')
    violinResiduals(mw.data, dataIDs, 'kalman-fs-control', 'chartrand-fs-control', 'butter-fs-control', 'Roach_theta')
    '''
    '''
    plotChartrand2('160403-chartrand-pa-control', 'b', 'chartrand-pa-control-1e-4', 'kalman-fs-control', 'Chartrand')
    plotChartrand2('160403-chartrand-pa-mass', 'r', 'chartrand-pa-mass-1e-4', 'kalman-fs-mass', 'Chartrand')
    plotChartrand2('160403-chartrand-pa-inertia', 'g', 'chartrand-pa-inertia-1e-4', 'kalman-fs-inertia', 'Chartrand')
    '''
    '''
    for alpha in range(10):
        plotChartrand2('220306-chartrand-fs-control-1e-' + str(alpha), 'b', 'chartrand-fs-control-1e-' + str(alpha), 'kalman-fs-control', 'Chartrand')
        plotChartrand2('220306-chartrand-fs-mass-1e-' + str(alpha), 'r', 'chartrand-fs-mass-1e-' + str(alpha), 'kalman-fs-mass', 'Chartrand')
        plotChartrand2('220306-chartrand-fs-inertia-1e-' + str(alpha), 'g', 'chartrand-fs-inertia-1e-' + str(alpha), 'kalman-fs-inertia', 'Chartrand')
    '''

    '''
    #for alpha in range(10):
    for alpha in [3]:
        alphas = [1.*(10**-alpha), 1.*(10**-alpha)]
        #print 'chartrand-fs-control-' + '1e-' + str(alpha)

        dataIDs = mo.treatments.query("Treatment == 'control'").index
        mw.csvLoadData(dataIDs)
        mw.data = fl.tarsusInCartFrame(mw.data, dataIDs)
        template = mc.jsonLoadTemplate('templateControl')
        generateEstimateTable('chartrandCart-pa-control-' + '1e-' + str(alpha), mw.data, dataIDs, samples, iterations, template, 'Chartrand', iterations=iterations, method='pa', alphas=alphas)

        dataIDs = mo.treatments.query("Treatment == 'mass'").index
        mw.csvLoadData(dataIDs)
        mw.data = fl.tarsusInCartFrame(mw.data, dataIDs)
        template = mc.jsonLoadTemplate('templateMass')
        generateEstimateTable('chartrandCart-pa-mass-' + '1e-' + str(alpha), mw.data, dataIDs, samples, iterations, template, 'Chartrand', iterations=iterations, method='pa', alphas=alphas)

        dataIDs = mo.treatments.query("Treatment == 'inertia'").index
        mw.csvLoadData(dataIDs)
        mw.data = fl.tarsusInCartFrame(mw.data, dataIDs)
        template = mc.jsonLoadTemplate('templateInertia')
        generateEstimateTable('chartrandCart-pa-inertia-' + '1e-' + str(alpha), mw.data, dataIDs, samples, iterations, template, 'Chartrand', iterations=iterations, method='pa', alphas=alphas)
    '''
