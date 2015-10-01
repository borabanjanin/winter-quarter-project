import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pickle
import scipy
from scipy import signal
from scipy import interpolate
import pandas as pd
import math
from collections import deque
import copy
from util import num
import kalman

def calcForces(data, template):
    xDiff = np.diff(np.diff(data['Roach_x']))
    yDiff = np.diff(np.diff(data['Roach_y']))
    rotDiff = np.diff(data['Roach_omega'])
    xDiff = list(xDiff)
    yDiff = list(yDiff)
    rotDiff = list(rotDiff)
    xDiff.insert(0,0)
    yDiff.insert(0,0)
    rotDiff.insert(0,0)
    xDiff.append(0)
    yDiff.append(0)
    rotDiff.append(0)

    data['xForce'] = 0
    data['yForce'] = 0
    data['torque'] = 0

    for i in data.index:
        accelVector = np.matrix([[xDiff[i]], [yDiff[i]], [0]])
        data.ix[i, 'xForce'] = accelVector.item(0,0) * template['m']
        data.ix[i, 'yForce'] = accelVector.item(1,0) * template['m']
        data.ix[i, 'torque'] = rotDiff[i] * template['I']

        pbc = np.matrix([[template['d'] * np.sin(data.ix[i, 'Roach_heading'] - np.pi/2), template['d'] * np.cos(data.ix[i, 'Roach_heading'] - np.pi/2), 0]])
        phat = np.matrix([[0, 0, pbc.item(0,1)], [0, 0, pbc.item(0,0)], [pbc.item(0,1), pbc.item(0,0), 0]])

        zeroMatrix = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ft = np.matrix([[data.ix[i, 'xForce']], [data.ix[i, 'yForce']], [0], [0], [0], [data.ix[i, 'torque']]])

        transformation = np.vstack([np.hstack([model.ModelWrapper.rotMat3(0).transpose(), \
        zeroMatrix]),np.hstack([-model.ModelWrapper.rotMat3(0).transpose()*phat, model.ModelWrapper.rotMat3(0).transpose()])])

        fc = transformation * ft

        data.ix[i, 'xForce'] = fc.item(0,0)
        data.ix[i, 'yForce'] = fc.item(1,0)
        data.ix[i, 'torque'] = fc.item(5,0)

    data.ix[0, 'xForce'] = np.nan
    data.ix[0, 'yForce'] = np.nan
    return data

def calcPolarForces(data, template):
    data['forceMagnitude'] = np.nan
    data['forceAngle'] = np.nan

    data['forceMagnitude'] = np.sqrt(data['xForce']**2 + data['yForce']**2)
    data['forceAngle'] = np.arctan2(data['yForce'], data['xForce']) - (data['Roach_theta'] % (2*np.pi))

def filterForces(data, dataID, template):
    #Forces take an index to calculate
    data['xForceSavgol'] = float('nan')
    data['yForceSavgol'] = float('nan')
    data['forceMagnitudeSavgol'] = float('nan')
    indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(data, 283, 'pre')
    indexStart += 1
    data['xForceSavgol'][indexStart:indexEnd] = signal.savgol_filter(data['Roach_x'][indexStart:indexEnd].dropna(), window_length=53, polyorder=3, deriv = 2) * template['m']
    data['yForceSavgol'][indexStart:indexEnd] = signal.savgol_filter(data['Roach_y'][indexStart:indexEnd].dropna(), window_length=53, polyorder=3, deriv = 2) * template['m']
    data['forceMagnitudeSavgol'] = np.sqrt(data['xForceSavgol']**2 + data['yForceSavgol']**2)
    return data

def findPhaseSegment(data, template):
    data['PhaseSegment'] = -1
    phaseSegment = 0
    indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(data, 283, 'pre')
    currentPhase = data.ix[indexStart, 'Roach_xv_phase']

    for i in range(indexStart,indexEnd):
        if data.ix[i, 'Roach_xv_phase'] % (2*np.pi) - np.pi < 0 and currentPhase % (2*np.pi) - np.pi > 0 \
        or  data.ix[i, 'Roach_xv_phase'] % (2*np.pi) - np.pi > 0 and currentPhase % (2*np.pi) - np.pi < 0:
            currentPhase = data.ix[i, 'Roach_xv_phase']
            phaseSegment += 1

        data.ix[i, 'PhaseSegment'] = phaseSegment

    return data

def plotFootData(dataID, template, tracker):
    mw.csvLoadData([dataID])
    data = mw.data[dataID]
    data = calcForces(data, template)
    data = filterForces(data, template)

    indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(data, 283, 'pre')
    data = findPhaseSegment(data, template)

    for segment in data['PhaseSegment'].unique():
        if segment == -1:
            continue
        plt.figure(segment, figsize=(12,8))
        plt.clf()
        tracker[dataID] = segment
        dataQuery = data.query('PhaseSegment == ' + str(segment))
        for i in dataQuery.index:
            x = dataQuery.ix[i, 'Roach_x']
            y = dataQuery.ix[i, 'Roach_y']
            xF = -dataQuery.ix[i, 'xForce']
            yF = -dataQuery.ix[i, 'yForce']
            xH = -template['d'] * np.sin(dataQuery.ix[i, 'Roach_heading'] - np.pi/2)
            yH = -template['d'] * np.cos(dataQuery.ix[i, 'Roach_heading'] - np.pi/2)
            #alpha = np.sqrt(LINE_LENGTH/(xF**2 + yF**2))
            alpha = 10
            plt.plot([x + xH, x + xF * alpha + xH], [y + yH, y + yF * alpha + yH], color='b', linestyle='-', linewidth=1)
            #plt.plot(x - template['d'] * np.sin(data.ix[i, 'Roach_heading'] - np.pi/2), y - template['d'] * np.cos(data.ix[i, 'Roach_heading'] - np.pi/2), marker='o', c='g')
            plt.plot(dataQuery.ix[i, 'Roach_x'], dataQuery.ix[i, 'Roach_y'], marker='.', c='r')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        #plt.title('Y Force', y=1.08)
        plt.grid(True)
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticks())
        plt.tight_layout()
        plt.savefig('Footlocation/' + str(dataID) + '-' + str(segment) + '.png')


        plt.figure(99, figsize=(12,8))
        plt.clf()
        plt.plot(data['yForce'][0:283].dropna(),'b')
        plt.xlabel('x (Sample)')
        plt.ylabel('y (Newtons?)')
        #plt.title('Y Force', y=1.08)
        plt.grid(True)
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticks())
        plt.tight_layout()
        plt.savefig('Footlocation/' + str(dataID) + '-Force.png')

        plt.figure(100, figsize=(12,8))
        plt.clf()
        plt.plot(scipy.fft(data['yForce'][75:283].dropna()),'b')
        plt.xlabel('x (Frequency)')
        plt.ylabel('y (Magnitude)')
        #plt.title('Y Force', y=1.08)
        plt.grid(True)
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticks())
        plt.tight_layout()
        plt.savefig('Footlocation/' + str(dataID) + '-FFT.png')

    return data

def phaseSamples(samples):
    sampleInterval = 2*np.pi/float(samples)
    phasePoints = [sampleInterval * i for i in range(samples)]
    phaseValues = {}
    for phasePoint in phasePoints:
        phaseValues[phasePoint] = deque()
    return sampleInterval, phasePoints, phaseValues

def interpolateSamples(samples, sampleInterval, data, dataID, column, phasePoints, phaseValues):
    data = data[dataID]
    indexStart, indexEnd = model.ModelWrapper.findDataIndexStatic(data, 283, 'pre')
    f = interpolate.interp1d(data['Roach_xv_phase'], data[column], bounds_error=False, fill_value=float('nan'))
    for i in range(indexStart+1, indexEnd+1):
        value = data.ix[i, column]
        phase = data.ix[i, 'Roach_xv_phase']
        phaseValue = phase % (2*np.pi)
        phasePoint = phaseValue - (phaseValue % sampleInterval)
        phasePointNext = phasePoints[(phasePoints.index(phasePoint)+1) % samples]
        phasePointDiff = phaseValue - phasePoint
        phasePointNextDiff = phaseValue - phasePointNext
        if phasePoint in phasePoints and phasePointNext in phasePoints:
            if math.isnan(f(phase + phasePointDiff)) == False:
                phaseValues[phasePoint].append(f(phase + phasePointDiff))
            if math.isnan(f(phase + phasePointNextDiff)) == False:
                phaseValues[phasePointNext].append(f(phase + phasePointNextDiff))
        else:
            raise Exception('Not finding sample index in dictionary keys')

    return phaseValues

def phaseStats(phaseValues):
    phaseGrid = deque()
    phaseMean = deque()
    phaseLQuantile = deque()
    phaseUQuantile = deque()
    #for key in sorted(phaseValues, key=lambda key: phaseValues[key]):
    for key in phaseValues.keys():
        phasePoint = pd.Series(phaseValues[key])

        if len(phasePoint) == 0:
            phaseGrid.append(key)
            phaseMean.append(pd.Series([np.nan]).mean())
            phaseLQuantile.append(pd.Series([np.nan]).mean())
            phaseUQuantile.append(pd.Series([np.nan]).mean())
            continue

        phaseGrid.append(key)
        phaseMean.append(pd.Series(phasePoint).mean())
        phaseLQuantile.append(pd.Series(phasePoint).quantile(q=[.025]).ix[.025])
        phaseUQuantile.append(pd.Series(phasePoint).quantile(q=[.975]).ix[.975])

    phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile = (list(t) for t in zip(*sorted(zip(phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile))))
    return phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile

def findPhaseValues(data, dataIDs, column):
    samples = 64
    sampleInterval, phasePoints, phaseValues = phaseSamples(samples)
    phaseValuesTotal = copy.deepcopy(phaseValues)
    for dataID in dataIDs:
        phaseValues = interpolateSamples(samples, sampleInterval, data, dataID, column, phasePoints, phaseValues)
        for key in phaseValues.keys():
            for value in phaseValues[key]:
                phaseValuesTotal[key].append(value)

    phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile = phaseStats(phaseValuesTotal)
    return (phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile)

def phasePlot(plotID, data, dataIDs, column):
    (phaseGrid, phaseMean, phaseLQuantile, phaseUQuantile) = findPhaseValues(data, dataIDs, column)
    phases = []
    variable = []

    plt.figure(plotID, figsize=(12,8))
    plt.clf()
    plt.plot(phaseGrid,phaseMean,'g-')
    plt.plot(phaseGrid,phaseLQuantile,'g--')
    plt.plot(phaseGrid,phaseUQuantile,'g--')
    plt.xlabel('x (Radians)')
    #plt.ylabel(column)
    plt.ylabel(column.replace('_',' '))
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.show()

    return phaseGrid, phaseMean

def findFootLocationData(data, dataIDs):
    tarsus1x = deque()
    tarsus2x = deque()
    tarsus3x = deque()
    tarsus4x = deque()
    tarsus5x = deque()
    tarsus6x = deque()
    tarsus1y = deque()
    tarsus2y = deque()
    tarsus3y = deque()
    tarsus4y = deque()
    tarsus5y = deque()
    tarsus6y = deque()
    rightStance = deque()
    for dataID in dataIDs:
        currentData = data[dataID]
        phase = currentData['Roach_xv_phase'].dropna()
        currentData['Roach_xv_phase'] =  currentData['Roach_xv_phase'] % (2 * np.pi)
        phaseIndex = phase.index
        #phaseIndex.pop(0)
        for i in phaseIndex[1:]:
            if currentData.ix[i, 'Roach_xv_phase'] < currentData.ix[i-1, 'Roach_xv_phase']:
                tarsus1x.append(currentData.ix[i, 'TarsusBody1_x'])
                tarsus3x.append(currentData.ix[i, 'TarsusBody3_x'])
                tarsus5x.append(currentData.ix[i, 'TarsusBody5_x'])
                tarsus1y.append(currentData.ix[i, 'TarsusBody1_y'])
                tarsus3y.append(currentData.ix[i, 'TarsusBody3_y'])
                tarsus5y.append(currentData.ix[i, 'TarsusBody5_y'])
            elif currentData.ix[i, 'Roach_xv_phase'] > np.pi and currentData.ix[i-1, 'Roach_xv_phase'] < np.pi:
                tarsus2x.append(currentData.ix[i, 'TarsusBody2_x'])
                tarsus4x.append(currentData.ix[i, 'TarsusBody4_x'])
                tarsus6x.append(currentData.ix[i, 'TarsusBody6_x'])
                tarsus2y.append(currentData.ix[i, 'TarsusBody2_y'])
                tarsus4y.append(currentData.ix[i, 'TarsusBody4_y'])
                tarsus6y.append(currentData.ix[i, 'TarsusBody6_y'])
        tarsus1x_mean = pd.Series(tarsus1x).mean()
        tarsus2x_mean = pd.Series(tarsus2x).mean()
        tarsus3x_mean = pd.Series(tarsus3x).mean()
        tarsus4x_mean = pd.Series(tarsus4x).mean()
        tarsus5x_mean = pd.Series(tarsus5x).mean()
        tarsus6x_mean = pd.Series(tarsus6x).mean()
        tarsus1y_mean = pd.Series(tarsus1y).mean()
        tarsus2y_mean = pd.Series(tarsus2y).mean()
        tarsus3y_mean = pd.Series(tarsus3y).mean()
        tarsus4y_mean = pd.Series(tarsus4y).mean()
        tarsus5y_mean = pd.Series(tarsus5y).mean()
        tarsus6y_mean = pd.Series(tarsus6y).mean()
        return (tarsus1x_mean, tarsus1y_mean), (tarsus3x_mean, tarsus3y_mean), (tarsus5x_mean, tarsus5y_mean)

def tarsusTouchDown(data, column):
    BUFFER = 18
    tm = num.localmin(BUFFER, np.asarray(data[column][0:284]))
    tM = num.localmin(BUFFER, -np.asarray(data[column][0:284]))
    t = np.zeros(800)
    m = list(tm.nonzero()[0])
    M = list(tM.nonzero()[0])
    if len(m) == 0 or len(M) == 0:
        return t, 0, 799
    minIndex = min(m[0],M[0])
    maxIndex = max(m[-1],M[-1])
    t[:minIndex] = np.nan
    t[maxIndex:] = np.nan
    indicator = np.nan

    if (m[0] < M[0] and len(M) > len(m)) or len(M) > len(m)+1:
        print 'undefined tarsus'
        return t, 0, 799
    if (M[0] < m[0] and len(m) > len(M)) or len(m) > len(M)+1:
        print 'undefined tarsus'
        return t, 0, 799



    for i in range(minIndex, maxIndex + 1):
        if m[0] - i < M[0] - i:
            t[i] = -1
            if M[0] - i == 0:
                m.pop(0)
                if indicator == 0:
                    raise Exception('Two consecutive minimums')
                indicator = 0
        elif M[0] - i < m[0] - i:
            t[i] = 1
            if m[0] - i == 0:
                M.pop(0)
                if indicator == 1:
                    raise Exception('Two consecutive maximums')
                indicator = 1
    return t, minIndex, maxIndex

def tripodStance(data):
    data['c0'] = np.nan
    data['c1'] = np.nan
    t1, mI1, MI1 = tarsusTouchDown(data, 'TarsusBody1_x')
    t2, mI2, MI2 = tarsusTouchDown(data, 'TarsusBody2_x')
    t3, mI3, MI3 = tarsusTouchDown(data, 'TarsusBody3_x')
    t4, mI4, MI4 = tarsusTouchDown(data, 'TarsusBody4_x')
    t5, mI5, MI5 = tarsusTouchDown(data, 'TarsusBody5_x')
    t6, mI6, MI6 = tarsusTouchDown(data, 'TarsusBody6_x')

    for i in data.index:
        if t1[i] == 1 and t3[i] == 1 and t5[i] == 1:
            data.ix[i, 'c0'] = 1
        elif t1[i] == -1 and t3[i] == -1 and t5[i] == -1:
            data.ix[i, 'c0'] = 0

    for i in data.index:
        if t2[i] == 1 and t4[i] == 1 and t6[i] == 1:
            data.ix[i, 'c1'] = 1
        elif t2[i] == -1 and t4[i] == -1 and t6[i] == -1:
            data.ix[i, 'c1'] = 0

    #testing code
    '''
    plt.figure(1); plt.clf();
    plt.plot(t1, 'b.')
    plt.plot(t1m.nonzero()[0],-1*np.ones(np.shape(t1m.nonzero()[0])),'go')
    plt.plot(t1M.nonzero()[0],np.ones(np.shape(t1M.nonzero()[0])),'ro')
    axes = plt.gca()
    axes.set_xlim([100,550])
    axes.set_ylim([-1.5,1.5])
    plt.show()
    '''
    return data

def averageTripodStance(data):
    data['c0'] = np.nan
    data['c1'] = np.nan
    t1, mI1, MI1 = tarsusTouchDown(data, 'TarsusBody1_x')
    t2, mI2, MI2 = tarsusTouchDown(data, 'TarsusBody2_x')
    t3, mI3, MI3 = tarsusTouchDown(data, 'TarsusBody3_x')
    t4, mI4, MI4 = tarsusTouchDown(data, 'TarsusBody4_x')
    t5, mI5, MI5 = tarsusTouchDown(data, 'TarsusBody5_x')
    t6, mI6, MI6 = tarsusTouchDown(data, 'TarsusBody6_x')

    minIndex = max(mI1, mI2, mI3, mI4, mI5, mI6)
    maxIndex = min(MI1, MI2, MI3, MI4, MI5, MI6)

    for i in range(minIndex, maxIndex + 1):
        if (t1[i] + t3[i] + t5[i]) > 1.5:
            data.ix[i, 'c0'] = 1
        else:
            data.ix[i, 'c0'] = 0

    for i in range(minIndex, maxIndex + 1):
        if (t2[i] + t4[i] + t6[i]) > 1.5:
            data.ix[i, 'c1'] = 1
        else:
            data.ix[i, 'c1'] = 0

    #testing code
    '''
    plt.figure(1); plt.clf();
    plt.plot(t1, 'b.')
    plt.plot(t1m.nonzero()[0],-1*np.ones(np.shape(t1m.nonzero()[0])),'go')
    plt.plot(t1M.nonzero()[0],np.ones(np.shape(t1M.nonzero()[0])),'ro')
    axes = plt.gca()
    axes.set_xlim([100,550])
    axes.set_ylim([-1.5,1.5])
    plt.show()
    '''
    return data, minIndex, maxIndex

def averageStance(data):
    data['c0'] = np.nan
    data['c1'] = np.nan

    data['TarsusBodyAvg1_x'] = (data['TarsusBody1_x'] + data['TarsusBody3_x'] + data['TarsusBody5_x'])/3.0
    data['TarsusBodyAvg1_y'] = (data['TarsusBody1_y'] + data['TarsusBody3_y'] + data['TarsusBody5_y'])/3.0
    data['TarsusBodyAvg2_x'] = (data['TarsusBody2_x'] + data['TarsusBody4_x'] + data['TarsusBody6_x'])/3.0
    data['TarsusBodyAvg2_y'] = (data['TarsusBody2_y'] + data['TarsusBody4_y'] + data['TarsusBody6_y'])/3.0

    t1, mI1, MI1 = tarsusTouchDown(data, 'TarsusBodyAvg1_x')
    t2, mI2, MI2 = tarsusTouchDown(data, 'TarsusBodyAvg2_x')

    minIndex = max(mI1, mI2)
    maxIndex = min(MI1, MI2)

    for i in range(minIndex, maxIndex + 1):
        if t1[i] == 1:
            data.ix[i, 'c0'] = 1
        else:
            data.ix[i, 'c0'] = 0

        if t2[i] == 1:
            data.ix[i, 'c1'] = 1
        else:
            data.ix[i, 'c1'] = 0

    #testing code
    '''
    plt.figure(1); plt.clf();
    plt.plot(t1, 'b.')
    plt.plot(t1m.nonzero()[0],-1*np.ones(np.shape(t1m.nonzero()[0])),'go')
    plt.plot(t1M.nonzero()[0],np.ones(np.shape(t1M.nonzero()[0])),'ro')
    axes = plt.gca()
    axes.set_xlim([100,550])
    axes.set_ylim([-1.5,1.5])
    plt.show()
    '''
    return data, minIndex, maxIndex

def findGaitCounts(data, beginIndex, endIndex):
    leftStance = 0
    rightStance = 0
    bothStance = 0
    aerialStance = 0

    for i in range(beginIndex, endIndex+1):
        if data.ix[i, 'c0'] == 1 and data.ix[i, 'c1'] == 1:
            bothStance += 1
        elif data.ix[i, 'c0'] == 1:
            leftStance += 1
        elif data.ix[i, 'c1'] == 1:
            rightStance += 1
        elif data.ix[i, 'c0'] == 0 and data.ix[i, 'c1'] == 0:
            aerialStance += 1

    return (leftStance, rightStance, bothStance, aerialStance)


def bootstrapGaitCounts(samples, countsList):
    meansList = []
    for i in range(samples):
        randomCounts = np.random.choice(range(len(countsList)), len(countsList))
        leftStanceSum = 0
        rightStanceSum = 0
        bothStanceSum = 0
        aerialStanceSum = 0
        for randomCount in randomCounts:
            (leftStance, rightStance, bothStance, aerialStance) = countsList[randomCount]
            leftStanceSum += leftStance
            rightStanceSum += rightStance
            bothStanceSum += bothStance
            aerialStanceSum += aerialStance
        totalSum = leftStanceSum + rightStanceSum + bothStanceSum + aerialStanceSum
        leftStanceMean = float(leftStanceSum)/totalSum
        rightStanceMean = float(rightStanceSum)/totalSum
        bothStanceMean = float(bothStanceSum)/totalSum
        aerialStanceMean = float(aerialStanceSum)/totalSum
        meansList.append((leftStanceMean, rightStanceMean, bothStanceMean, aerialStanceMean))
    leftStanceSeries = pd.Series(zip(*meansList)[0])
    rightStanceSeries = pd.Series(zip(*meansList)[1])
    bothStanceSeries = pd.Series(zip(*meansList)[2])
    aerialStanceSeries = pd.Series(zip(*meansList)[3])
    leftStats = (leftStanceSeries.quantile(q=[.01]).ix[.01], leftStanceSeries.mean(), leftStanceSeries.quantile(q=[.99]).ix[.99])
    rightStats = (rightStanceSeries.quantile(q=[.01]).ix[.01], rightStanceSeries.mean(), rightStanceSeries.quantile(q=[.99]).ix[.99])
    bothStats = (bothStanceSeries.quantile(q=[.01]).ix[.01], bothStanceSeries.mean(), bothStanceSeries.quantile(q=[.99]).ix[.99])
    aerialStats = (aerialStanceSeries.quantile(q=[.01]).ix[.01], aerialStanceSeries.mean(), aerialStanceSeries.quantile(q=[.99]).ix[.99])
    return (leftStats, rightStats, bothStats, aerialStats)

def findGaitStatsTrial(samples, data, dataIDs):
    countsList = []
    for dataID in dataIDs:
        currentData = data[dataID]
        currentData = tripodStance(currentData)
        countsList.append(findGaitCounts(currentData, 0, 284))
    stats = bootstrapGaitCounts(samples, countsList)
    print 'Left Stance: ' + str(stats[0])
    print 'Right Stance: ' + str(stats[1])
    print 'Ground Stance: ' + str(stats[2])
    print 'Aerial Stance: ' + str(stats[3])

def findSteps(data, endIndex):
    data['Roach_xv_phase'] = data['Roach_xv_phase'] % (2 * np.pi)
    beginIndex = data['Roach_xv_phase'].first_valid_index()
    currentPhase = data.ix[beginIndex, 'Roach_xv_phase']
    transitionPoints = []
    for i in range(beginIndex, endIndex):
        if data.ix[i, 'Roach_xv_phase'] < currentPhase:
            transitionPoints.append(i)
        currentPhase = data.ix[i, 'Roach_xv_phase']
    return transitionPoints

def findStepsAlt(data, beginIndex, endIndex):
    data['Roach_xv_phase'] = data['Roach_xv_phase'] % (2 * np.pi)
    currentPhase = data.ix[beginIndex, 'Roach_xv_phase']
    transitionPoints = []
    for i in range(beginIndex, endIndex):
        if data.ix[i, 'Roach_xv_phase'] < currentPhase:
            transitionPoints.append(i)
        currentPhase = data.ix[i, 'Roach_xv_phase']
    return transitionPoints

def findGaitStatsStep(samples, data, dataIDs):
    countsList = []
    for dataID in dataIDs:
        currentData = data[dataID]
        currentData = tripodStance(currentData)
        transitionPoints = findSteps(currentData, 284)
        for i in range(len(transitionPoints) - 1):
            countsList.append(findGaitCounts(currentData, transitionPoints[i], transitionPoints[i+1]))
            #if findGaitCounts(currentData, transitionPoints[i], transitionPoints[i+1])[0] == 0:
            #    print dataID
            #    print transitionPoints[i]
            #    print transitionPoints[i+1]
    stats = bootstrapGaitCounts(samples, countsList)
    print 'Left Stance: ' + str(stats[0])
    print 'Right Stance: ' + str(stats[1])
    print 'Ground Stance: ' + str(stats[2])
    print 'Aerial Stance: ' + str(stats[3])

def findGaitStatsStepAlt(samples, data, dataIDs):
    countsList = []
    for dataID in dataIDs:
        currentData = data[dataID]
        currentData, beginIndex, endIndex = averageStance(currentData)
        transitionPoints = findStepsAlt(currentData, beginIndex, 284)
        for i in range(len(transitionPoints) - 1):
            countsList.append(findGaitCounts(currentData, transitionPoints[i], transitionPoints[i+1]))
            #if findGaitCounts(currentData, transitionPoints[i], transitionPoints[i+1])[0] == 0:
            #    print dataID
            #    print transitionPoints[i]
            #    print transitionPoints[i+1]
    stats = bootstrapGaitCounts(samples, countsList)
    print 'Left Stance: ' + str(stats[0])
    print 'Right Stance: ' + str(stats[1])
    print 'Ground Stance: ' + str(stats[2])
    print 'Aerial Stance: ' + str(stats[3])

def stepFinder(data, minIndex, maxIndex):
    stepsList = []
    currentStep = []

    for i in range(minIndex, maxIndex + 1):
        if data.ix[i, 'StanceIndicator'] == 1:
            currentStep.append(i)
        else:
            if len(currentStep) > 0:
                stepsList.append((currentStep[0], currentStep[-1]))
                currentStep = []
    if len(stepsList) > 0 and currentStep != stepsList[-1] and len(currentStep) > 0:
        stepsList.append((currentStep[0], currentStep[-1]))
    return stepsList

def stanceData(data, stance):
    data, minIndex, maxIndex = averageStance(data)
    data['StanceIndicator'] = 0

    if stance == 'left':
        for i in range(minIndex, maxIndex + 1):
            if data.ix[i, 'c0'] == 1 and data.ix[i, 'c1'] == 0:
                data.ix[i, 'StanceIndicator'] = 1
    elif stance == 'right':
        for i in range(minIndex, maxIndex + 1):
            if data.ix[i, 'c0'] == 0 and data.ix[i, 'c1'] == 1:
                data.ix[i, 'StanceIndicator'] = 1
    elif stance == 'ground':
        for i in range(minIndex, maxIndex + 1):
            if data.ix[i, 'c0'] == 1 and data.ix[i, 'c1'] == 1:
                data.ix[i, 'StanceIndicator'] = 1
    elif stance == 'aerial':
        for i in range(minIndex, maxIndex + 1):
            if data.ix[i, 'c0'] == 0 and data.ix[i, 'c1'] == 0:
                data.ix[i, 'StanceIndicator'] = 1
    else:
        raise Exception('Not valid stance flag')

    stepsList = stepFinder(data, minIndex, 283)
    #del data['StanceIndicator']

    return stepsList

def findStanceSteps(data, dataIDs, stance):
    stepsList = {}
    for dataID in dataIDs:
        currentData = data[dataID]
        stepsList[dataID] = stanceData(currentData, stance)
    return stepsList

def stanceOrientation(data, steps, columnListPostion, columnListHeading):
    stances = []
    for step in steps:
        xPos = deque()
        yPos = deque()
        xHead = deque()
        yHead = deque()
        zBody = data['Roach_x'][step[0]:step[1]+1] + 1.j* data['Roach_y'][step[0]:step[1]+1]
        for column in columnListPostion:
            #xPos.extendleft(deque(data[column+'_x'][step[0]:step[1]+1]))
            #yPos.extendleft(deque(data[column+'_y'][step[0]:step[1]+1]))
            zFoot = data[column+'_x'][step[0]:step[1]+1] + 1.j* data[column+'_y'][step[0]:step[1]+1]
            zFootTransform = zBody + zFoot * np.exp(1.j * data['Roach_theta'][step[0]:step[1]+1])
            xPos.extendleft(deque(zFootTransform.real))
            yPos.extendleft(deque(zFootTransform.imag))
        for column in columnListHeading:
            zFoot = data[column+'_x'][step[0]:step[1]+1] + 1.j* data[column+'_y'][step[0]:step[1]+1]
            zFootTransform = zBody + zFoot * np.exp(1.j * data['Roach_theta'][step[0]:step[1]+1])
            xHead.extendleft(deque(zFootTransform.real))
            yHead.extendleft(deque(zFootTransform.imag))

        xy = np.matrix([xHead, yHead])
        U, s, V = np.linalg.svd(xy)
        xPosAvg = pd.Series(xPos).mean()
        yPosAvg = pd.Series(yPos).mean()
        thetaPosAvg = np.arctan2(U.item(1,0), U.item(0,0))
        if abs(thetaPosAvg) > np.pi/2:
            thetaPosAvg = np.arctan2(-1.0 * U.item(1,0), -1.0 * U.item(0,0))
        stances.append((xPosAvg, yPosAvg, thetaPosAvg))

    return stances

def findStanceOrientation(data, stepsList, stance):
    stanceList = {}
    columnListPosition = []
    columnListHeading = []

    if stance == 'left':
        columnListPosition = ['TarsusBodyAvg1']
        columnListHeading = ['TarsusBody1', 'TarsusBody3', 'TarsusBody5']
    elif stance == 'right':
        columnListPosition = ['TarsusBodyAvg2']
        columnListHeading = ['TarsusBody2', 'TarsusBody4', 'TarsusBody6']
    elif stance == 'ground':
        columnListPosition = ['TarsusBodyAvg1', 'TarsusBodyAvg2']
        columnListHeading = ['TarsusBody1', 'TarsusBody2', 'TarsusBody3', \
        'TarsusBody4', 'TarsusBody5', 'TarsusBody6']
    elif stance == 'aerial':
        columnListPosition = ['TarsusBodyAvg1', 'TarsusBodyAvg2']
        columnListHeading = ['TarsusBody1', 'TarsusBody2', 'TarsusBody3', \
        'TarsusBody4', 'TarsusBody5', 'TarsusBody6']
    else:
        raise Exception('Not valid stance flag')

    for dataID in stepsList.keys():
        currentData = data[dataID]
        stanceList[dataID] = stanceOrientation(currentData, stepsList[dataID], columnListPosition, columnListHeading)

    return stanceList

def createPWHamilInput(data, dataIDs, template, obsID, stance):
    stepDict = findStanceSteps(data, dataIDs, stance)
    stanceDict = findStanceOrientation(data, stepDict, stance)
    qList = deque()
    accelList = deque()

    tarsusStepList = deque()

    for dataID in dataIDs:
        currentData = data[dataID]
        kalman.dataEstimates(data, dataID, template, obsID)

    for dataID in dataIDs:
        cData = data[dataID]
        stepList = stepDict[dataID]
        stanceList = stanceDict[dataID]
        for cStep, cStance in zip(stepList, stanceList):
            zBody = cData['Roach_x_KalmanDDX'][cStep[0]:cStep[1]+1] + 1.j * cData['Roach_y_KalmanDDX'][cStep[0]:cStep[1]+1]
            zTransform = zBody * np.exp(1.j * -cStance[2])
            cData['Roach_x_KalmanDDX'][cStep[0]:cStep[1]+1] = zTransform.real
            cData['Roach_y_KalmanDDX'][cStep[0]:cStep[1]+1] = zTransform.imag
            for i in range(cStep[0], cStep[1]+1):
                qList.append((cData.ix[i, 'Roach_x'], cData.ix[i, 'Roach_y'], cData.ix[i, 'Roach_theta']))
                accelList.append((cData.ix[i, 'Roach_x_KalmanDDX'], cData.ix[i, 'Roach_y_KalmanDDX'], cData.ix[i, 'Roach_theta_KalmanDDX']))
            tarsusList = deque()
            tarsusList.append(cData['TarsusBody1_x'][cStep[0]:cStep[1]+1])
            tarsusList.append(cData['TarsusBody2_x'][cStep[0]:cStep[1]+1])
            tarsusList.append(cData['TarsusBody3_x'][cStep[0]:cStep[1]+1])
            tarsusList.append(cData['TarsusBody4_x'][cStep[0]:cStep[1]+1])
            tarsusList.append(cData['TarsusBody5_x'][cStep[0]:cStep[1]+1])
            tarsusList.append(cData['TarsusBody6_x'][cStep[0]:cStep[1]+1])
            tarsusStepList.append(tarsusList)

    return qList, accelList, stepDict, stanceDict, tarsusStepList


'''
def findGaitStats(data, dataIDs):

    for data in dataIDs:
        currentData = data[dataID]
        currentData = tripodStance(currentData)
        (leftStance, rightStance, bothStance, aerialStance) = findGaitCountsTrial(data)
    ts = float(leftStance + rightStance + bothStance + aerialStance)
    return (leftStance/ts, rightStance/ts, bothStance/ts, aerialStance/ts, ts)
'''

if __name__ == "__main__":
    saveDir = 'StableOrbit'

    varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t']

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)

    LINE_LENGTH = 8

    #step data
    '''
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    template = mc.jsonLoadTemplate('templateControl')
    mw.csvLoadData(dataIDs)
    qList, accelList = createPWHamilInput(mw.data, [0], template, 'left')
    '''

    #gait statistics
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    #findGaitStatsTrial(3, mw.data, dataIDs)
    #findGaitStatsStep(3, mw.data, dataIDs)
    #findGaitStatsStepAlt(3, mw.data, dataIDs)

    print 'control'
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    findGaitStatsStepAlt(10, mw.data, dataIDs)

    print 'mass'
    #undefined tarsus dataIDs: 79, 82, 92
    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    mw.csvLoadData(dataIDs)
    findGaitStatsStepAlt(10, mw.data, dataIDs)

    print 'inertia'
    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    mw.csvLoadData(dataIDs)
    findGaitStatsStepAlt(10, mw.data, dataIDs)

    #something
    '''
    tracker = {}
    template = mc.jsonLoadTemplate('templateControl')
    template['d'] = 0
    dataIDs = mo.treatments.query("Treatment == 'control' and AnimalID == 8").index
    for dataID in dataIDs:
        mw.csvLoadData([dataID])
        calcForces(mw.data[dataID], template)
        calcPolarForces(mw.data[dataID], template)
        filterForces(mw.data[dataID], dataID, template)
        #data = plotFootData(dataID, template, tracker)
    '''

    #phaseGrid, phaseMean = phasePlot(99, mw.data, dataIDs, 'forceMagnitude')
    #phaseValues, phaseSamplesCount = phasePlot(100, mw.data, dataIDs, 'forceAngle')
    #phaseValues, phaseSamplesCount = phasePlot(101, mw.data, dataIDs, 'forceMagnitudeSavgol')
    #phaseValues, phaseSamplesCount = phasePlot(101, mw.data, dataIDs, 'xForce')
    #phaseValues, phaseSamplesCount = phasePlot(102, mw.data, dataIDs, 'yForce')
    #phaseValues, phaseSamplesCount = phasePlot(103, mw.data, dataIDs, 'xForceSavgol')
    #phaseValues, phaseSamplesCount = phasePlot(104, mw.data, dataIDs, 'yForceSavgol')
    #phaseValues, phaseSamplesCount = phasePlot(105, mw.data, dataIDs, 'Roach_vx')
    #phaseValues, phaseSamplesCount = phasePlot(105, mw.data, dataIDs, 'Roach_vy')
    #phaseValues, phaseSamplesCount = phasePlot(105, mw.data, dataIDs, 'Roach_heading')
    #phaseValues, phaseSamplesCount = phasePlot(105, mw.data, dataIDs, 'Roach_theta')

    #phaseValues, phaseSamplesCount = phasePlot(200, mw.data, dataIDs, 'TarsusCombined_x')
    #phaseValues, phaseSamplesCount = phasePlot(200, mw.data, dataIDs, 'TarsusBody3_x')
    #phaseValues, phaseSamplesCount = phasePlot(201, mw.data, dataIDs, 'TarsusBody3_vx')
    #phaseValues, phaseSamplesCount = phasePlot(202, mw.data, dataIDs, 'Roach_pitch')
    #phaseValues, phaseSamplesCount = phasePlot(203, mw.data, dataIDs, 'Roach_roll')
    #phaseValues, phaseSamplesCount = phasePlot(204, mw.data, dataIDs, 'Roach_yaw')
    #phaseValues, phaseSamplesCount = phasePlot(205, mw.data, dataIDs, 'Roach_v')

    #with open(os.path.join(os.getcwd(),'Footlocation','plots.p'), 'wb') as f:
    #    pickle.dump(tracker, f)


    #run footlocation
    #d = mw.data[15]
    #plt.plot(np.asarray(d['TarsusBody1_x']))
    #x = np.asarray(d['TarsusBody1_x'])
    #from util import num
    #m = num.localmin(15, x)
    #plt.plot(m,x[m],'go')
    #m
    #plt.plot(m.nonzero()[0],x[m == 0.],'go')
    #m.nonzero()[0].shape
    #x[m == 0.]
    #x[m > 0]
    #plt.plot(m.nonzero()[0],x[m > 0.],'go')
    #m = num.localmin(10, x)
    #plt.plot(m.nonzero()[0],x[m > 0.],'rx')
    #M = num.localmin(10, -x)
    #plt.plot(M.nonzero()[0],x[M > 0.],'rx')
