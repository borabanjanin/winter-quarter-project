import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd

def phaseDiff(data, index0, index1):
    if np.isnan(data.ix[index0, 'Roach_xv_phase']):
        print dataID
    return data.ix[index1, 'Roach_xv_phase'] - data.ix[index0, 'Roach_xv_phase']

def findPhaseDiff(data, offset, numPeriods):
    index = offset - 1
    while(phaseDiff(data, index, offset) < numPeriods * 2.0*np.pi):
        index -= 1
    return offset - index

def findPeriod(data, offset, dt, numPeriods):
    periodSamples = findPhaseDiff(data, offset, numPeriods)
    return (periodSamples * dt)/numPeriods

def findAverageSpeed(data, offset, numPeriods):
    totalSpeed = 0.0
    periodSamples = findPhaseDiff(data, offset, numPeriods)
    r = range(offset-periodSamples, offset+1)
    for i in r:
        if np.isnan(data.ix[i, 'Roach_v']) == False:
            totalSpeed += data.ix[i, 'Roach_v']
    return totalSpeed/len(r)

saveDir = 'StableOrbit'
offset = 283

varList = ['t','x','y','theta','fx','fy','dx','dy','dtheta','q']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

mw.csvLoadData(mo.treatments.index)

dataStats = pd.DataFrame(columns=('AverageSpeed','Frequency','StrideLength'))

numPeriods = 2
for dataID in mo.treatments.index:
    period = findPeriod(mw.data[dataID], offset, .002, numPeriods)
    avgSpeed = findAverageSpeed(mw.data[dataID], offset, numPeriods)
    dataStats.loc[dataID] = [avgSpeed, 1/period, avgSpeed*period]

obsIDs = [0,1,2]
mw.csvLoadObs(obsIDs)
for obsID in obsIDs:
    period = mw.findPeriod(obsID) * .0005
    speed = mw.findSpeed(obsID) /period
    print 'Speed: ' + str(speed)
    print 'Frequency: ' + str(1/period)
    print 'Stride Length: ' + str(speed * period)
    #Calculated Stride Length:2.1612092234725591

controlDataIDs = mo.treatments.query("Treatment == 'control'").index
c = dataStats.index.map(lambda x: x in controlDataIDs)
dataStats[c]['AverageSpeed'].mean()
dataStats[c]['Frequency'].mean()
dataStats[c]['StrideLength'].mean()

inertiaDataIDs = mo.treatments.query("Treatment == 'inertia'").index
c = dataStats.index.map(lambda x: x in controlDataIDs)
dataStats[c]['AverageSpeed'].mean()
dataStats[c]['Frequency'].mean()
dataStats[c]['StrideLength'].mean()

massDataIDs = mo.treatments.query("Treatment == 'mass'").index
c = dataStats.index.map(lambda x: x in controlDataIDs)
dataStats[c]['AverageSpeed'].mean()
dataStats[c]['Frequency'].mean()
dataStats[c]['StrideLength'].mean()

dataStats.to_csv(os.path.join(os.getcwd(),'populationstats2.csv'), sep='\t')

plt.clf()
plt.figure(1)
plt.plot(mw.observations[0]['x'],mw.observations[0]['y'],'b')
plt.plot(mw.observations[1]['x'],mw.observations[1]['y'],'g')
plt.plot(mw.observations[2]['x'],mw.observations[2]['y'],'r')


'''
plt.clf()
plt.figure(1)
plt.plot(mw.observations[0]['x'],mw.observations[0]['y'])
plt.plot(mw.observations[1]['x'],mw.observations[1]['y'])
plt.plot(mw.observations[2]['x'],mw.observations[2]['y'])

templateControl = mc.jsonLoadTemplate('templateControl')
templateInertia = mc.jsonLoadTemplate('templateInertia')
templateMass = mc.jsonLoadTemplate('templateMass')
templateControl['t'] = 3.0
templateInertia['t'] = 3.0
templateMass['t'] = 3.0
templateControl['dt'] = 0.0005
templateInertia['dt'] = 0.0005
templateMass['dt'] = 0.0005
obsIDs = []
obsIDs.append(mw.runTrial('lls.LLS', templateControl, varList, 0))
obsIDs.append(mw.runTrial('lls.LLS', templateInertia, varList, 0))
obsIDs.append(mw.runTrial('lls.LLS', templateMass, varList, 0))
mw.saveTables()
'''
