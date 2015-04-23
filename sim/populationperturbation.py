import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd
import copy


saveDir = 'test'
offset = 283

varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

mw.csvLoadData(mo.treatments.index)

mwOrbit = model.ModelWrapper('StableOrbit')
mcOrbit = model.ModelConfiguration(mwOrbit)
mwOrbit.csvLoadObs([0,1,2])

'''
controlObsIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
mw.csvLoadObs(controlObsIDs)
massObsIDs = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
mw.csvLoadObs(massObsIDs)
inertiaObsIDs = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124]
mw.csvLoadObs(inertiaObsIDs)


dataStatsControl = pd.DataFrame(columns=('x','y'))
controlDataIDs = mo.treatments.query("Treatment == 'control'").index
dataControl = mp.combineData(controlDataIDs, offset)
c = dataControl['DataID'].map(lambda x: x in controlDataIDs)
for sampleID in [int(sampleID) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]:
    d = dataControl['SampleID'].map(lambda x: x == sampleID)
    dataStatsControl.loc[sampleID] = [dataControl[c][d]['Roach_x'].mean(), dataControl[c][d]['Roach_y'].mean()]

obsStatsControl = pd.DataFrame(columns=('x','y'))
obsControl = mp.combineObs(controlObsIDs, 0)
for sampleID in [int(sampleID) for sampleID in obsControl['SampleID'].unique()]:
    d = obsControl['SampleID'].map(lambda x: x == sampleID)
    obsStatsControl.loc[sampleID] = [obsControl[d]['x'].mean(), obsControl[d]['y'].mean()]

plt.clf()
plt.figure(1)
dataLegend, = plt.plot(dataStatsControl['x'],dataStatsControl['y'],'b',label='Data')
obsLegend, = plt.plot(obsStatsControl['x'],obsStatsControl['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x')
plt.ylabel('y')
plt.title('Control Treatment with No Phase: x vs y', y=1.08)
plt.show()

plt.clf()
plt.figure(2)
timeData = [round(sampleID * .002,4) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsControl['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsControl['x'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsControl['x'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('x')
plt.title('Control Treatment with No Phase: t vs x', y=1.08)
plt.show()

plt.clf()
plt.figure(3)
timeData = [round(sampleID * .002,4) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsControl['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsControl['y'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsControl['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('y')
plt.title('Control Treatment with No Phase: t vs y', y=1.08)
plt.show()

dataStatsMass = pd.DataFrame(columns=('x','y'))
massDataIDs = mo.treatments.query("Treatment == 'mass'").index
dataMass = mp.combineData(massDataIDs, offset)
c = dataMass['DataID'].map(lambda x: x in massDataIDs)
for sampleID in [int(sampleID) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]:
    d = dataMass['SampleID'].map(lambda x: x == sampleID)
    dataStatsMass.loc[sampleID] = [dataMass[c][d]['Roach_x'].mean(), dataMass[c][d]['Roach_y'].mean()]

obsStatsMass = pd.DataFrame(columns=('x','y'))
obsMass = mp.combineObs(massObsIDs, 0)
for sampleID in [int(sampleID) for sampleID in obsMass['SampleID'].unique()]:
    d = obsMass['SampleID'].map(lambda x: x == sampleID)
    obsStatsMass.loc[sampleID] = [obsMass[d]['x'].mean(), obsMass[d]['y'].mean()]

plt.clf()
plt.figure(4)
dataLegend, = plt.plot(dataStatsMass['x'],dataStatsMass['y'],'b',label='Data')
obsLegend, = plt.plot(obsStatsMass['x'],obsStatsMass['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mass Treatment with No Phase: x vs y', y=1.08)
plt.show()

plt.clf()
plt.figure(5)
timeData = [round(sampleID * .002,4) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsMass['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsMass['x'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsMass['x'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('x')
plt.title('Mass Treatment with No Phase: t vs x', y=1.08)
plt.show()

plt.clf()
plt.figure(6)
timeData = [round(sampleID * .002,4) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsMass['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsMass['y'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsMass['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('y')
plt.title('Mass Treatment with No Phase: t vs y', y=1.08)
plt.show()

dataStatsInertia = pd.DataFrame(columns=('x','y'))
inertiaDataIDs = mo.treatments.query("Treatment == 'inertia'").index
dataInertia = mp.combineData(inertiaDataIDs, offset)
c = dataInertia['DataID'].map(lambda x: x in inertiaDataIDs)
for sampleID in [int(sampleID) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]:
    d = dataInertia['SampleID'].map(lambda x: x == sampleID)
    dataStatsInertia.loc[sampleID] = [dataInertia[c][d]['Roach_x'].mean(), dataInertia[c][d]['Roach_y'].mean()]

obsStatsInertia = pd.DataFrame(columns=('x','y'))
obsInertia = mp.combineObs(inertiaObsIDs, 0)
for sampleID in [int(sampleID) for sampleID in obsInertia['SampleID'].unique()]:
    d = obsInertia['SampleID'].map(lambda x: x == sampleID)
    obsStatsInertia.loc[sampleID] = [obsInertia[d]['x'].mean(), obsInertia[d]['y'].mean()]

plt.clf()
plt.figure(7)
dataLegend, = plt.plot(dataStatsInertia['x'],dataStatsInertia['y'],'b',label='Data')
obsLegend, = plt.plot(obsStatsInertia['x'],obsStatsInertia['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x')
plt.ylabel('y')
plt.title('Inertia Treatment with No Phase: x vs y', y=1.08)
plt.show()

plt.clf()
plt.figure(8)
timeData = [round(sampleID * .002,4) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsInertia['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsInertia['x'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsInertia['x'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('x')
plt.title('Inertia Treatment with No Phase: t vs x', y=1.08)
plt.show()

plt.clf()
plt.figure(9)
timeData = [round(sampleID * .002,4) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsInertia['SampleID'].unique() if sampleID >= 0]
dataLegend, = plt.plot(timeData,dataStatsInertia['y'],'b',label='Data')
obsLegend, = plt.plot(timeObs,obsStatsInertia['y'],'g',label='Observation')
plt.legend(handles=[dataLegend, obsLegend], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t')
plt.ylabel('y')
plt.title('Inertia Treatment with No Phase: t vs y', y=1.08)
plt.show()
'''


def findPhaseData(data, sampleIndex):
    return data.ix[sampleIndex, 'Roach_xv_phase'] % (2 * np.pi)

def matchPhase(obsID,phaseData):
    (beginInex, endIndex) = mwOrbit.findObsIndex(obsID)
    stableOrbitPeriod  = mwOrbit.findPeriod(obsID)
    bestIndex = endIndex
    bestPhaseDiff = 2.0 * np.pi
    for i in range(stableOrbitPeriod):
        phaseObs  = mwOrbit.findPhase(endIndex - i, obsID)
        if np.abs(phaseObs - phaseData) < bestPhaseDiff:
            bestIndex = endIndex - i
            bestPhaseDiff = np.abs(phaseObs - phaseData)
    return bestIndex

testObsIDs = []

#0 ID is the control observation
controlObsID = 0
controlObsIDs = []
controlDataIDs = mo.treatments.query("Treatment == 'control'").index
for dataID in controlDataIDs:
    (beginIndex, endIndex) = mw.findDataIndex(dataID, offset, 'post')
    template = mcOrbit.jsonLoadTemplate('templateControl')
    template = mc.setRunTime(beginIndex, endIndex, template)
    template = mc.setConfValuesData(offset, dataID, template, ['x', 'y', 'theta'])
    samplePhaseData = findPhaseData(mw.data[dataID], offset)/(2 * np.pi)
    bestPhaseIndex = matchPhase(controlObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(controlObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(controlObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]

    oID = mw.runTrial('lls.LLS', template, varList, dataID, 'dataAccel')
    tID = mw.runTrial('lls.LLS', template, varList, dataID)

    print 'expected length: ' + str((template['tf'] - template['to'])/template['dt'])
    print 'actual length test: ' + str(len(mw.observations[tID].index))
    print 'actual length pert: ' + str(len(mw.observations[oID].index))
    print ''

    #controlObsIDs.append(oID)
    testObsIDs.append(tID)

#mw.saveTables()

'''
#1 ID is the mass observation
massObsID = 1
massObsIDs = []
massDataIDs = mo.treatments.query("Treatment == 'mass'").index
for dataID in massDataIDs:
    (beginIndex, endIndex) = mw.findDataIndex(dataID, offset, 'post')
    template = mcOrbit.jsonLoadTemplate('templateMass')
    template = mc.setRunTime(beginIndex, endIndex, template)
    template = mc.setConfValuesData(offset, dataID, template, ['x','y','theta'])
    samplePhaseData = findPhaseData(mw.data[dataID], offset)/(2 * np.pi)
    bestPhaseIndex = matchPhase(massObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(massObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(massObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]
    massObsIDs.append(mw.runTrial('lls.LLS', template, varList, dataID, 'dataAccel'))

#2 ID is the inertia observation
inertiaObsID = 2
inertiaObsIDs = []
inertiaDataIDs = mo.treatments.query("Treatment == 'inertia'").index
for dataID in inertiaDataIDs:
    (beginIndex, endIndex) = mw.findDataIndex(dataID, offset, 'post')
    template = mcOrbit.jsonLoadTemplate('templateInertia')
    template = mc.setRunTime(beginIndex, endIndex, template)
    template = mc.setConfValuesData(offset, dataID, template, ['x','y','theta'])
    samplePhaseData = findPhaseData(mw.data[dataID], offset)/(2 * np.pi)
    bestPhaseIndex = matchPhase(inertiaObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(inertiaObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(inertiaObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]
    inertiaObsIDs.append(mw.runTrial('lls.LLS', template, varList, dataID, 'dataAccel'))


print 'controlObsIDs: ' + str(controlObsIDs)
print 'massObsIDs: ' + str(massObsIDs)
print 'inertiaObsIDs: ' + str(inertiaObsIDs)


mw.saveTables()
'''
