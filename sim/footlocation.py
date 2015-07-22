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

if __name__ == "__main__":
    saveDir = 'StableOrbit'

    varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t']

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)

    LINE_LENGTH = 8

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
    phaseValues, phaseSamplesCount = phasePlot(200, mw.data, dataIDs, 'TarsusBody3_x')
    phaseValues, phaseSamplesCount = phasePlot(201, mw.data, dataIDs, 'TarsusBody3_vx')
    phaseValues, phaseSamplesCount = phasePlot(202, mw.data, dataIDs, 'Roach_pitch')
    phaseValues, phaseSamplesCount = phasePlot(203, mw.data, dataIDs, 'Roach_roll')
    phaseValues, phaseSamplesCount = phasePlot(204, mw.data, dataIDs, 'Roach_yaw')
    phaseValues, phaseSamplesCount = phasePlot(205, mw.data, dataIDs, 'Roach_v')



    #with open(os.path.join(os.getcwd(),'Footlocation','plots.p'), 'wb') as f:
    #    pickle.dump(tracker, f)
