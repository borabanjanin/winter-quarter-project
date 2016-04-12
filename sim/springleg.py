import numpy as np
import pylab as plt
from util import poly
import sympy
import string
import footlocation as fl
import modelwrapper as model
import pickle
import os
import kalman
import pandas as pd
from collections import deque
import copy
import pwhamil as pw
import random
from shrevz import util as shutil

#Global Parameters
LEGS = 6
GRID_SAMPLES = 64
ORDER = 3

def fit(q,a,dv,si,m_=None,m=None):
    """
    c = fit(q,a,dv)  fit potential to accel data using least-squares
    i.e. if a = c dV(q) where dV(q) = sum([dv[j](q) for j in range(len(dv))])
    then c = a^T X (X^T X)^{-1} where X = dV(q)

    Inputs:
        q - N x Nd - N samples of Nd dimensional configurations
        a - N x Nd - N samples of Nd dimensional accelerations
        dv - Nb - derivatives of potential basis functions
       dv[j] : Nd -> Nd
       (optional)
       m_ - Nd x Nd - (constant) mass matrix / inertia tensor
       m : Nd -> Nd x Nd - (configuration-dependent) mass matrix / inertia tensor

    Outputs:
        c - Nb x Nd - coefficients for Nb potential basis functions in Nd variables
    """
    #print 'q' + str(np.shape(q))
    #print 'a' + str(np.shape(a))
    #print 'si' + str(np.shape(si))

    # q, a contains N observations of Nd dimensional configurations
    N, Nd = a.shape
    b = len(dv)

    # form composite force vector and derivative matrix
    F = []; D = []
    F = a.flatten()

    D = np.zeros((N*Nd, b*LEGS))

    for i in range(N):
        for j in range(LEGS):
            D[i*Nd:(i+1)*Nd,j*b:(j+1)*b] = si[i, j] * np.array([dvb(*q[i, j:j+Nd]) for dvb in dv]).T

    # solve D c = F for potential basis function coefficients c

    D = np.matrix(D)
    F = np.matrix(F)
    c = np.linalg.inv(D.T*D)*D.T*F.T
    return c

def stanceInfo(data, dataIDs, xh, yh, template, obsID, label, hip = None):
    results = {}
    q = deque()
    a = deque()
    si = deque()
    bodyEstimates = deque()
    timeSeriesData = pd.DataFrame()

    kp = kalman.kalmanParameters(template, obsID)
    kalman.dataEstimates2(data, dataIDs, kp)

    for dataID in dataIDs:
        currentData = data[dataID]
        currentData = fl.tripodStance(currentData, tarsusInfo = True)

        currentData['t1'][currentData[currentData['t1'] == 0].index] = np.nan
        currentData['t2'][currentData[currentData['t2'] == 0].index] = np.nan
        currentData['t3'][currentData[currentData['t3'] == 0].index] = np.nan
        currentData['t4'][currentData[currentData['t4'] == 0].index] = np.nan
        currentData['t5'][currentData[currentData['t5'] == 0].index] = np.nan
        currentData['t6'][currentData[currentData['t6'] == 0].index] = np.nan

        currentData['t1'][currentData[currentData['t1'] == -1].index] = 0
        currentData['t2'][currentData[currentData['t2'] == -1].index] = 0
        currentData['t3'][currentData[currentData['t3'] == -1].index] = 0
        currentData['t4'][currentData[currentData['t4'] == -1].index] = 0
        currentData['t5'][currentData[currentData['t5'] == -1].index] = 0
        currentData['t6'][currentData[currentData['t6'] == -1].index] = 0

        #find indexes where all legs are defined
        minIndex = max(currentData['t1'].first_valid_index()
            ,currentData['t2'].first_valid_index()
            ,currentData['t3'].first_valid_index()
            ,currentData['t4'].first_valid_index()
            ,currentData['t5'].first_valid_index()
            ,currentData['t6'].first_valid_index())
        maxIndex = min(currentData['t1'].last_valid_index()
            ,currentData['t2'].last_valid_index()
            ,currentData['t3'].last_valid_index()
            ,currentData['t4'].last_valid_index()
            ,currentData['t5'].last_valid_index()
            ,currentData['t6'].last_valid_index())

        if maxIndex > 283:
            maxIndex = 283

        if minIndex == None or maxIndex == None:
            continue

        if hip == None:
            x1, y1, theta1 = legtoHip(currentData['TarsusBody1_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody1_y_KalmanX'][minIndex:maxIndex+1], xh[0], yh[0])
            x2, y2, theta2 = legtoHip(currentData['TarsusBody2_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody2_y_KalmanX'][minIndex:maxIndex+1], xh[1], yh[1])
            x3, y3, theta3 = legtoHip(currentData['TarsusBody3_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody3_y_KalmanX'][minIndex:maxIndex+1], xh[2], yh[2])
            x4, y4, theta4 = legtoHip(currentData['TarsusBody4_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody4_y_KalmanX'][minIndex:maxIndex+1], xh[3], yh[3])
            x5, y5, theta5 = legtoHip(currentData['TarsusBody5_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody5_y_KalmanX'][minIndex:maxIndex+1], xh[4], yh[4])
            x6, y6, theta6 = legtoHip(currentData['TarsusBody6_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody6_y_KalmanX'][minIndex:maxIndex+1], xh[5], yh[5])

        elif hip == 'NoHip':
            x1, y1, theta1 = legtoCOM(currentData['TarsusBody1_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody1_y_KalmanX'][minIndex:maxIndex+1])
            x2, y2, theta2 = legtoCOM(currentData['TarsusBody2_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody2_y_KalmanX'][minIndex:maxIndex+1])
            x3, y3, theta3 = legtoCOM(currentData['TarsusBody3_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody3_y_KalmanX'][minIndex:maxIndex+1])
            x4, y4, theta4 = legtoCOM(currentData['TarsusBody4_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody4_y_KalmanX'][minIndex:maxIndex+1])
            x5, y5, theta5 = legtoCOM(currentData['TarsusBody5_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody5_y_KalmanX'][minIndex:maxIndex+1])
            x6, y6, theta6 = legtoCOM(currentData['TarsusBody6_x_KalmanX'][minIndex:maxIndex+1], currentData['TarsusBody6_y_KalmanX'][minIndex:maxIndex+1])

        Roach_x = currentData['Roach_x_KalmanX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_y = currentData['Roach_y_KalmanX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_theta = currentData['Roach_theta_KalmanX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_ddx = currentData['Roach_x_KalmanDDX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_ddy = currentData['Roach_y_KalmanDDX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_ddtheta = currentData['Roach_theta_KalmanDDX'][minIndex:maxIndex+1].astype(np.float64)
        Roach_phase = (currentData['Roach_xv_phase'][minIndex:maxIndex+1].astype(np.float64) % 2*np.pi)
        Roach_c0 = (currentData['c0'][minIndex:maxIndex+1].astype(np.float64) % 2*np.pi)
        Roach_c1 = (currentData['c1'][minIndex:maxIndex+1].astype(np.float64) % 2*np.pi)
        dataIDColumn = dataID * np.ones(maxIndex+1-minIndex)

        q.append(np.vstack((x1, y1, theta1, x2, y2, theta2, x3, y3, theta3, x4, y4, theta4, x5, y5, theta5, x6, y6, theta6)).T)
        a.append(np.vstack((currentData['Roach_x_KalmanDDX'][minIndex:maxIndex+1], currentData['Roach_y_KalmanDDX'][minIndex:maxIndex+1], currentData['Roach_theta_KalmanDDX'][minIndex:maxIndex+1])).T)
        bodyEstimates.append(np.vstack((Roach_x, Roach_y, Roach_theta, Roach_ddx, Roach_ddy, Roach_ddtheta, Roach_phase, Roach_c0, Roach_c1, dataIDColumn)).T)
        si.append(np.vstack((
        currentData['t1'][minIndex:maxIndex+1]
        ,currentData['t2'][minIndex:maxIndex+1]
        ,currentData['t3'][minIndex:maxIndex+1]
        ,currentData['t4'][minIndex:maxIndex+1]
        ,currentData['t5'][minIndex:maxIndex+1]
        ,currentData['t6'][minIndex:maxIndex+1]
        )).T)

    if len(q) == 0:
        print 'Warning: No StanceInfo saved'
        return

    bodyArray = np.vstack(bodyEstimates)
    timeSeriesData['Roach_x_KalmanX'] = bodyArray[:,0]
    timeSeriesData['Roach_y_KalmanX'] = bodyArray[:,1]
    timeSeriesData['Roach_theta_KalmanX'] = bodyArray[:,2]
    timeSeriesData['Roach_x_KalmanDDX'] = bodyArray[:,3]
    timeSeriesData['Roach_y_KalmanDDX'] = bodyArray[:,4]
    timeSeriesData['Roach_theta_KalmanDDX'] = bodyArray[:,5]
    timeSeriesData['Roach_xv_phase'] = bodyArray[:,6]
    timeSeriesData['c0'] = bodyArray[:,7]
    timeSeriesData['c1'] = bodyArray[:,8]
    timeSeriesData['dataID'] = bodyArray[:,9]

    qArray = np.vstack(q)
    aArray = np.vstack(a)
    stanceIndicatorTable = np.vstack(si)
    results['qArray'] = qArray
    results['aArray'] = aArray
    results['stanceIndicatorTable'] = stanceIndicatorTable
    results['timeSeriesTable'] = timeSeriesData
    #results['timeSeriesData'] = data
    pickle.dump(results, open(os.path.join(os.getcwd(), 'StanceInfo',label+'.p'), 'wb'))
    return data

def legtoHip(x, y, xh, yh, thetaOffset=None):
    x_h = x - xh
    y_h = y - yh
    z_h = x_h.astype(np.float64) + 1.j * y_h.astype(np.float64)
    if thetaOffset != None:
        z_h = z_h * np.exp(1.j * thetaOffset)
    theta_h = np.arctan2(z_h.real.astype(np.float64), z_h.imag.astype(np.float64))
    #if thetaOffset != None:
    #    theta_h = theta_h + thetaOffset
    return x_h, y_h, theta_h

def legtoCOM(x, y, thetaOffset=None):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    theta = np.arctan2(x.astype(np.float64), y.astype(np.float64))
    return x, y, theta

def fourierSeriesStanceInfo(fourierSeriesLabel, label, xh, yh):
    results = {}
    fsTable = loadFourierSeries(fourierSeriesLabel)
    x1, y1, theta1 = legtoHip(fsTable['TarsusBody1_x_KalmanX'], fsTable['TarsusBody1_y_KalmanX'], xh[0], yh[0])
    x2, y2, theta2 = legtoHip(fsTable['TarsusBody2_x_KalmanX'], fsTable['TarsusBody2_y_KalmanX'], xh[1], yh[1])
    x3, y3, theta3 = legtoHip(fsTable['TarsusBody3_x_KalmanX'], fsTable['TarsusBody3_y_KalmanX'], xh[2], yh[2])
    x4, y4, theta4 = legtoHip(fsTable['TarsusBody4_x_KalmanX'], fsTable['TarsusBody4_y_KalmanX'], xh[3], yh[3])
    x5, y5, theta5 = legtoHip(fsTable['TarsusBody5_x_KalmanX'], fsTable['TarsusBody5_y_KalmanX'], xh[4], yh[4])
    x6, y6, theta6 = legtoHip(fsTable['TarsusBody6_x_KalmanX'], fsTable['TarsusBody6_y_KalmanX'], xh[5], yh[5])

    q = deque()
    a= deque()

    q.append(np.vstack((x1, y1, theta1, x2, y2, theta2, x3, y3, theta3, x4, y4, theta4, x5, y5, theta5, x6, y6, theta6)).T)
    a.append(np.vstack((fsTable['Roach_x_KalmanDDX'], fsTable['Roach_y_KalmanDDX'], fsTable['Roach_theta_KalmanDDX'])).T)

    iterations = len(fsTable.index)/64
    stanceIndicatorTable = np.zeros((len(fsTable.index),6))
    for i in range(iterations):
        stanceIndicatorTable[0+64*i:32+64*i,0] = 1
        stanceIndicatorTable[0+64*i:32+64*i,2] = 1
        stanceIndicatorTable[0+64*i:32+64*i,4] = 1
        stanceIndicatorTable[0+64*i:32+64*i,1] = 0
        stanceIndicatorTable[0+64*i:32+64*i,3] = 0
        stanceIndicatorTable[0+64*i:32+64*i,5] = 0
        stanceIndicatorTable[32+64*i:64+64*i,0] = 0
        stanceIndicatorTable[32+64*i:64+64*i,2] = 0
        stanceIndicatorTable[32+64*i:64+64*i,4] = 0
        stanceIndicatorTable[32+64*i:64+64*i,1] = 1
        stanceIndicatorTable[32+64*i:64+64*i,3] = 1
        stanceIndicatorTable[32+64*i:64+64*i,5] = 1

    qArray = np.vstack(q)
    aArray = np.vstack(a)
    results['qArray'] = qArray
    results['aArray'] = aArray
    results['stanceIndicatorTable'] = stanceIndicatorTable
    pickle.dump(results, open(os.path.join(os.getcwd(), 'StanceInfo',label+'.p'), 'wb'))

def estimatesTableStanceInfo(estimatesLabel, label, xh, yh, angleCorrection=np.zeros(6), hip=None):
    estimatesTable = pw.loadEstimatesTable(estimatesLabel)
    results = {}

    if hip == None:
        x1, y1, theta1 = legtoHip(estimatesTable['TarsusBody1_x_KalmanX_Mean'], estimatesTable['TarsusBody1_y_KalmanX_Mean'], xh[0], yh[0], angleCorrection[0])
        x2, y2, theta2 = legtoHip(estimatesTable['TarsusBody2_x_KalmanX_Mean'], estimatesTable['TarsusBody2_y_KalmanX_Mean'], xh[1], yh[1], angleCorrection[1])
        x3, y3, theta3 = legtoHip(estimatesTable['TarsusBody3_x_KalmanX_Mean'], estimatesTable['TarsusBody3_y_KalmanX_Mean'], xh[2], yh[2], angleCorrection[2])
        x4, y4, theta4 = legtoHip(estimatesTable['TarsusBody4_x_KalmanX_Mean'], estimatesTable['TarsusBody4_y_KalmanX_Mean'], xh[3], yh[3], angleCorrection[3])
        x5, y5, theta5 = legtoHip(estimatesTable['TarsusBody5_x_KalmanX_Mean'], estimatesTable['TarsusBody5_y_KalmanX_Mean'], xh[4], yh[4], angleCorrection[4])
        x6, y6, theta6 = legtoHip(estimatesTable['TarsusBody6_x_KalmanX_Mean'], estimatesTable['TarsusBody6_y_KalmanX_Mean'], xh[5], yh[5], angleCorrection[5])
    elif hip == 'NoHip':
        x1, y1, theta1 = legtoCOM(estimatesTable['TarsusBody1_x_KalmanX_Mean'], estimatesTable['TarsusBody1_y_KalmanX_Mean'], angleCorrection[0])
        x2, y2, theta2 = legtoCOM(estimatesTable['TarsusBody2_x_KalmanX_Mean'], estimatesTable['TarsusBody2_y_KalmanX_Mean'], angleCorrection[1])
        x3, y3, theta3 = legtoCOM(estimatesTable['TarsusBody3_x_KalmanX_Mean'], estimatesTable['TarsusBody3_y_KalmanX_Mean'], angleCorrection[2])
        x4, y4, theta4 = legtoCOM(estimatesTable['TarsusBody4_x_KalmanX_Mean'], estimatesTable['TarsusBody4_y_KalmanX_Mean'], angleCorrection[3])
        x5, y5, theta5 = legtoCOM(estimatesTable['TarsusBody5_x_KalmanX_Mean'], estimatesTable['TarsusBody5_y_KalmanX_Mean'], angleCorrection[4])
        x6, y6, theta6 = legtoCOM(estimatesTable['TarsusBody6_x_KalmanX_Mean'], estimatesTable['TarsusBody6_y_KalmanX_Mean'], angleCorrection[5])

    qArray = np.vstack((x1, y1, theta1, x2, y2, theta2, x3, y3, theta3, x4, y4, theta4, x5, y5, theta5, x6, y6, theta6)).T

    aArray = np.vstack((estimatesTable['Roach_x_KalmanDDX_Mean'].astype(np.float64) \
    ,estimatesTable['Roach_y_KalmanDDX_Mean'].astype(np.float64) \
    ,estimatesTable['Roach_theta_KalmanDDX_Mean'].astype(np.float64))).T

    stanceIndicatorTable = np.zeros((64,6))
    stanceIndicatorTable[0:32,0] = 1
    stanceIndicatorTable[0:32,2] = 1
    stanceIndicatorTable[0:32,4] = 1
    stanceIndicatorTable[0:32,1] = 0
    stanceIndicatorTable[0:32,3] = 0
    stanceIndicatorTable[0:32,5] = 0
    stanceIndicatorTable[32:64,0] = 0
    stanceIndicatorTable[32:64,2] = 0
    stanceIndicatorTable[32:64,4] = 0
    stanceIndicatorTable[32:64,1] = 1
    stanceIndicatorTable[32:64,3] = 1
    stanceIndicatorTable[32:64,5] = 1

    results['stanceIndicatorTable'] = stanceIndicatorTable
    results['qArray'] = qArray
    results['aArray'] = aArray
    pickle.dump(results, open(os.path.join(os.getcwd(), 'StanceInfo',label+'.p'), 'wb'))

    return results

def runSpringLeg(stanceLabel, springLegLabel):
    results = {}
    stanceInfo = pw.loadStanceInfo(stanceLabel)

    results = {}
    V, variables = pw.potentialFunction(3, 1, 2)
    #print evaluatePotentialFunction(V, variables, [5, 2, 3])
    dV = pw.gradientFunctions(V, variables)
    lList = pw.gradientFunctionsToLambdas(dV, variables)

    c = fit(stanceInfo['qArray'], stanceInfo['aArray'], lList, stanceInfo['stanceIndicatorTable'])
    results['c'] = c
    results['dV'] = dV
    results['variables'] = variables

    pickle.dump(results, open(os.path.join(os.getcwd(), 'SpringLeg',springLegLabel+'.p'), 'wb'))
    return c

def evaluateGradientFunction(dv, C, q, si):
    #number of observations, number of dimensions
    N, Nd = np.shape(q)[0], np.shape(q)[1]/LEGS

    accs = np.zeros((N, Nd))
    b = len(dv)

    for i in range(N):
        for j in range(LEGS):
            accs[i,:] += np.sum(si[i, j] * np.array(C[j:j+b,0]) * np.array([dvb(*q[i, j:j+Nd]) \
            for dvb in dv]), axis=0)
    return accs

def runTimeSeriesPredictions(stanceLabel, springLegLabel):
    springLegResults = loadSpringLeg(springLegLabel)
    stanceInfo = pw.loadStanceInfo(stanceLabel)
    lList = pw.gradientFunctionsToLambdas(springLegResults['dV'], springLegResults['variables'])

    tsTable = stanceInfo['timeSeriesTable']

    accs = evaluateGradientFunction(lList, springLegResults['c'], stanceInfo['qArray'], stanceInfo['stanceIndicatorTable'])

    phaseBins = np.linspace(0, 2*np.pi, num=65)
    inds = np.digitize(tsTable['Roach_xv_phase'], phaseBins) - 1
    phaseAveraged = np.zeros((64, 7))
    stanceAveraged = np.zeros((64, 2))
    for i in tsTable.index:
        phaseBin = inds[i]
        phaseAveraged[phaseBin, 0] += 1
        phaseAveraged[phaseBin, 1] += tsTable.ix[i, 'Roach_x_KalmanDDX']
        phaseAveraged[phaseBin, 2] += tsTable.ix[i, 'Roach_y_KalmanDDX']
        phaseAveraged[phaseBin, 3] += tsTable.ix[i, 'Roach_theta_KalmanDDX']
        phaseAveraged[phaseBin, 4] += accs[i, 0]
        phaseAveraged[phaseBin, 5] += accs[i, 1]
        phaseAveraged[phaseBin, 6] += accs[i, 2]
        if ~np.isnan(tsTable.ix[i, 'c0']):
            stanceAveraged[phaseBin, 0] += 1
        if ~np.isnan(tsTable.ix[i, 'c1']):
            stanceAveraged[phaseBin, 1] += 1

    for i in range(64):
        phaseAveraged[i,1:7] = phaseAveraged[i,1:7] / phaseAveraged[i,0]
        P = stanceAveraged[i, 0] + stanceAveraged[i, 1]
        stanceAveraged[i, 0] = float(stanceAveraged[i, 0])/P
        stanceAveraged[i, 1] = float(stanceAveraged[i, 1])/P

    predictions = {}
    for dataID in tsTable['dataID'].unique():
        indexes = tsTable[tsTable['dataID'] == dataID].index
        modelAcc = evaluateGradientFunction(lList, springLegResults['c'], stanceInfo['qArray'][indexes], stanceInfo['stanceIndicatorTable'][indexes])
        kalmanAcc = np.vstack((tsTable['Roach_x_KalmanDDX'][indexes].astype(np.float64)
            ,tsTable['Roach_y_KalmanDDX'][indexes].astype(np.float64)
            ,tsTable['Roach_theta_KalmanDDX'][indexes].astype(np.float64))).T
        modelAcc = np.hstack((np.array(tsTable['Roach_xv_phase'][indexes][:,np.newaxis]), modelAcc))
        predictions[dataID] = np.hstack((modelAcc, kalmanAcc))
    plt.show()

    '''
    plt.figure(1)
    plt.plot(accs[:,0], '.')
    plt.plot(tsTable['Roach_x_KalmanDDX'], '.')
    plt.figure(2)
    plt.plot(accs[:,1], '.')
    plt.plot(tsTable['Roach_y_KalmanDDX'], '.')
    plt.figure(3)
    plt.plot(accs[:,2], '.')
    plt.plot(tsTable['Roach_theta_KalmanDDX'], '.')
    plt.show()

    plt.figure(1)
    plt.plot(phaseAveraged[:,1], 'g')
    plt.plot(phaseAveraged[:,4], 'b')
    plt.figure(2)
    plt.plot(phaseAveraged[:,2], 'g')
    plt.plot(phaseAveraged[:,5], 'b')
    plt.figure(3)
    plt.plot(phaseAveraged[:,3], 'g')
    plt.plot(phaseAveraged[:,6], 'b')
    plt.show()

    plt.figure(1)
    plt.plot(stanceAveraged[:,0], 'g')
    plt.plot(stanceAveraged[:,1], 'b')
    plt.show()
    '''
    return predictions

def bootstrapTimeSeriesPredictions(predictions, predictionIDs, samples):
    '''
    prediction:

    [    :        :             :              :             :              :            :         ]
    [phase, modelXAccel, modelYAccel, modelThetaAccel, kalmanXAccel, kalmanYAccel, kalmanThetaAccel]
    [    :        :             :              :             :              :            :         ]
    '''
    random.seed(2)
    mX = deque()
    mY = deque()
    mTheta = deque()
    kX = deque()
    kY = deque()
    kTheta = deque()
    for i in range(samples):
        randomDataIDs = np.random.choice(predictionIDs, len(predictionIDs))
        accels = deque()
        for dataID in randomDataIDs:
            accels.append(copy.deepcopy(predictions[dataID]))
        accelArray = np.vstack(accels)
        fsMX = shutil.FourierSeries()
        fsMY = shutil.FourierSeries()
        fsMTheta = shutil.FourierSeries()
        fsMX.fit(ORDER, accelArray[:,0][np.newaxis,:], accelArray[:,1][np.newaxis,:])
        fsMY.fit(ORDER, accelArray[:,0][np.newaxis,:], accelArray[:,2][np.newaxis,:])
        fsMTheta.fit(ORDER, accelArray[:,0][np.newaxis,:], accelArray[:,3][np.newaxis,:])
        grid = np.arange(GRID_SAMPLES)[np.newaxis,:] * (2*np.pi)/GRID_SAMPLES
        mX.append(fsMX.val(grid)[:,0].real)
        mY.append(fsMY.val(grid)[:,0].real)
        mTheta.append(fsMTheta.val(grid)[:,0].real)

        fsKX = shutil.FourierSeries()
        fsKY = shutil.FourierSeries()
        fsKTheta = shutil.FourierSeries()
        fsKX.fit(ORDER, copy.deepcopy(accelArray[:,0][np.newaxis,:]), accelArray[:,4][np.newaxis,:])
        fsKY.fit(ORDER, copy.deepcopy(accelArray[:,0][np.newaxis,:]), accelArray[:,5][np.newaxis,:])
        fsKTheta.fit(ORDER, copy.deepcopy(accelArray[:,0][np.newaxis,:]), accelArray[:,6][np.newaxis,:])
        grid = np.arange(GRID_SAMPLES)[np.newaxis,:] * (2*np.pi)/GRID_SAMPLES
        kX.append(fsKX.val(grid)[:,0].real)
        kY.append(fsKY.val(grid)[:,0].real)
        kTheta.append(fsKTheta.val(grid)[:,0].real)

    predictionTable = pd.DataFrame()

    mX = np.vstack(mX)
    mY = np.vstack(mY)
    mTheta = np.vstack(mTheta)
    kX = np.vstack(kX)
    kY = np.vstack(kY)
    kTheta = np.vstack(kTheta)
    predictionTable['Model_DDX_Mean'] = np.mean(mX, axis=0)
    predictionTable['Model_DDY_Mean'] = np.mean(mY, axis=0)
    predictionTable['Model_DDTheta_Mean'] = np.mean(mTheta, axis=0)
    predictionTable['Kalman_DDX_Mean'] = np.mean(kX, axis=0)
    predictionTable['Kalman_DDY_Mean'] = np.mean(kY, axis=0)
    predictionTable['Kalman_DDTheta_Mean'] = np.mean(kTheta, axis=0)
    predictionTable['Model_DDX_01'] = np.percentile(mX, 01, axis=0)
    predictionTable['Model_DDY_01'] = np.percentile(mY, 01, axis=0)
    predictionTable['Model_DDTheta_01'] = np.percentile(mTheta, 01, axis=0)
    predictionTable['Kalman_DDX_01'] = np.percentile(kX, 01, axis=0)
    predictionTable['Kalman_DDY_01'] = np.percentile(kY, 01, axis=0)
    predictionTable['Kalman_DDTheta_01'] = np.percentile(kTheta, 01, axis=0)
    predictionTable['Model_DDX_99'] = np.percentile(mX, 99, axis=0)
    predictionTable['Model_DDY_99'] = np.percentile(mY, 99, axis=0)
    predictionTable['Model_DDTheta_99'] = np.percentile(mTheta, 99, axis=0)
    predictionTable['Kalman_DDX_99'] = np.percentile(kX, 99, axis=0)
    predictionTable['Kalman_DDY_99'] = np.percentile(kY, 99, axis=0)
    predictionTable['Kalman_DDTheta_99'] = np.percentile(kTheta, 99, axis=0)

    plt.figure(1)
    #plt.plot(predictionTable['Model_DDX_Mean'])
    plt.plot(predictionTable['Kalman_DDX_Mean'])
    plt.figure(2)
    #plt.plot(predictionTable['Model_DDY_Mean'])
    plt.plot(predictionTable['Kalman_DDY_Mean'])
    plt.figure(3)
    #plt.plot(predictionTable['Model_DDTheta_Mean'])
    plt.plot(predictionTable['Kalman_DDTheta_Mean'])
    plt.show()

def runSpringLegEstimates(stanceLabel, estimatesLabel, springLegLabel):
    results = {}
    stanceInfo = pw.loadStanceInfo(stanceLabel)
    estimatesTable = pw.loadEstimatesTable(estimatesLabel)

    results = {}
    V, variables = pw.potentialFunction(3, 1, 2)
    #print evaluatePotentialFunction(V, variables, [5, 2, 3])
    dV = pw.gradientFunctions(V, variables)
    lList = pw.gradientFunctionsToLambdas(dV, variables)

    c = fit(stanceInfo['qArray'], stanceInfo['aArray'], lList, stanceInfo['stanceIndicatorTable'])
    results['c'] = c
    results['dV'] = dV
    results['variables'] = variables

    pickle.dump(results, open(os.path.join(os.getcwd(), 'SpringLeg',springLegLabel+'.p'), 'wb'))
    return c

def loadSpringLeg(label):
    return pickle.load(open(os.path.join(os.getcwd(), 'SpringLeg',label+'.p'), 'rb'))

def loadFourierSeries(label):
    return pickle.load(open(os.path.join(os.getcwd(), 'FourierSeries',label+'.p'), 'rb'))

def generateSpringLegTableEstimates(stanceLabel, springLegLabel, estimatesLabel):
    springLegResults = loadSpringLeg(springLegLabel)
    stanceInfo = pw.loadStanceInfo(stanceLabel)
    et = pw.loadEstimatesTable(estimatesLabel)
    lList = pw.gradientFunctionsToLambdas(springLegResults['dV'], springLegResults['variables'])

    springLegTable = pd.DataFrame(index=range(64), columns=['DDX', 'DDY', 'DDTheta'])

    accs = evaluateGradientFunction(lList, springLegResults['c'], stanceInfo['qArray'], stanceInfo['stanceIndicatorTable'])

    springLegTable['DDX'] = accs[:,0]
    springLegTable['DDY'] = accs[:,1]
    springLegTable['DDTheta'] = accs[:,2]

    plt.figure(1)
    plt.plot(springLegTable['DDX'])
    plt.plot(et['Roach_x_KalmanDDX_Mean'])
    plt.figure(2)
    plt.plot(springLegTable['DDY'])
    plt.plot(et['Roach_y_KalmanDDX_Mean'])
    plt.figure(3)
    plt.plot(springLegTable['DDTheta'])
    plt.plot(et['Roach_theta_KalmanDDX_Mean'])
    plt.show()

    return accs

if __name__ == "__main__":
    saveDir = 'StableOrbit'

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)
    dataIDs = [0, 1, 2, 3]
    mw.csvLoadData(dataIDs)
    mw.csvLoadObs([0,1,2])

    #physically realistic hip values
    xh = [1.8, 0.9 , 0., 0., 0.9, 1.8]
    yh = [1., 1., 1., -1., -1., -1.]

    samples = 50

    template = mc.jsonLoadTemplate('templateControl')

    #dataIDs = mo.treatments.query("Treatment == 'control'").index
    #mw.csvLoadData(dataIDs)
    #stanceInfo(mw.data, dataIDs, xh, yh, template, 0, 'test')#, hip='NoHip')
    #runTimeSeriesPredictions('test', '150201-springleg-data-control')

    #estimatesTableStanceInfo('control-raw', '150201-springleg-estimates-control', xh, yh, np.array([0, 0, 0, 0, 0, 1.5]), hip='NoHip')
    #estimatesTableStanceInfo('mass-estimates', '150201-springleg-estimates-mass', xh, yh, np.array([-1.46, 0, 0, 0, 0, 0.0]))
    #estimatesTableStanceInfo('inertia-estimates', '150201-springleg-estimates-inertia', xh, yh, np.array([0, 0, 0, 0, 0, -1.5]))
    #runSpringLegEstimates('control-estimates','250115-springleg-estimates-control')

    #fourierSeriesStanceInfo('test', 'test', xh, yh)
    #runSpringLeg('test', 'test')
    #generateSpringLegTable('test', 'test', 'control-estimates')

    #control predictions
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)

    stanceInfo(mw.data, dataIDs, xh, yh, template, 0, '150201-springleg-data-control')#, hip='NoHip')
    runSpringLeg('150201-springleg-data-control', '150201-springleg-data-control')
    #generateSpringLegTable('150201-springleg-estimates-control', '150201-springleg-data-control', 'control-estimates')
    predictions = runTimeSeriesPredictions('150201-springleg-data-control', '150201-springleg-data-control')
    bootstrapTimeSeriesPredictions(predictions, predictions.keys(), samples)

    kp = kalman.kalmanParameters(template, 0)
    kalman.dataEstimates2(mw.data, dataIDs, kp)

    predictions2 = {}
    for dataID in dataIDs:
        cd = mw.data[dataID]
        a = np.vstack((cd['Roach_xv_phase'][0:284].dropna() \
        ,cd['Roach_x_KalmanDDX'][0:284].dropna() \
        ,cd['Roach_y_KalmanDDX'][0:284].dropna() \
        ,cd['Roach_theta_KalmanDDX'][0:284].dropna() \
        ,cd['Roach_x_KalmanDDX'][0:284].dropna() \
        ,cd['Roach_y_KalmanDDX'][0:284].dropna() \
        ,cd['Roach_theta_KalmanDDX'][0:284].dropna()))
        predictions2[dataID] = a.T

    bootstrapTimeSeriesPredictions(predictions2, predictions2.keys(), samples)

    '''
    means = kalman.kalmanBootstrap(mw.data, dataIDs, samples, 'Roach_x')
    meanStatsX = kalman.kalmanMeanStats(means, 'Roach_x')
    means = kalman.kalmanBootstrap(mw.data, dataIDs, samples, 'Roach_y')
    meanStatsY = kalman.kalmanMeanStats(means, 'Roach_y')
    means = kalman.kalmanBootstrap(mw.data, dataIDs, samples, 'Roach_theta')
    meanStatsTheta = kalman.kalmanMeanStats(means, 'Roach_theta')
    '''
    #mass predictions
    #dataIDs = mo.treatments.query("Treatment == 'mass'").index
    #mw.csvLoadData(dataIDs)
    #stanceInfo(mw.data, dataIDs, xh, yh, template, 0, '150201-springleg-data-mass')
    #runSpringLeg('150201-springleg-data-mass', '150201-springleg-data-mass')
    #generateSpringLegTable('150201-springleg-estimates-mass', '150201-springleg-data-mass', 'mass-estimates')
    #predictions = runTimeSeriesPredictions('150201-springleg-data-mass', '150201-springleg-data-mass')
    #bootstrapTimeSeriesPredictions(predictions, predictions.keys(), samples)

    #inertia predictions
    #dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    #mw.csvLoadData(dataIDs)
    #stanceInfo(mw.data, dataIDs, xh, yh, template, 0, '150201-springleg-data-inertia')
    #runSpringLeg('150201-springleg-data-inertia', '150201-springleg-data-inertia')
    #generateSpringLegTable('150201-springleg-estimates-inertia', '150201-springleg-data-inertia', 'inertia-estimates')
    #predictions = runTimeSeriesPredictions('150201-springleg-data-inertia', '150201-springleg-data-inertia')
    #bootstrapTimeSeriesPredictions(predictions, predictions.keys(), samples)
