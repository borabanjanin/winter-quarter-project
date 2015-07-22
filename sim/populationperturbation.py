import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd
import random


saveDir = 'PerturbationPhasePuck'
offset = 283

varList = ['t','x','y','theta','fx','fy','dx','dy','dtheta','q','v','delta','omega','KE','accy']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

mw.csvLoadData(mo.treatments.index)

mwOrbit = model.ModelWrapper('StableOrbit')
mcOrbit = model.ModelConfiguration(mwOrbit)
mwOrbit.csvLoadObs([0,1,2])

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

#0 ID is the control observation
controlObsID = 0
controlObsIDs = []
controlDataIDs = mo.treatments.query("Treatment == 'control'").index
for dataID in controlDataIDs:
    (beginIndex, endIndex) = mw.findDataIndex(dataID, offset, 'post')
    template = mcOrbit.jsonLoadTemplate('templateControl')
    template = mc.setRunTime(beginIndex, endIndex, template)
    template = mc.setConfValuesData(offset, dataID, template, ['x','y','theta'])

    samplePhaseData = findPhaseData(mw.data[dataID], offset)/(2 * np.pi)
    #samplePhaseData = random.uniform(0,1)
    #samplePhaseData = 0.0
    bestPhaseIndex = matchPhase(controlObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(controlObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(controlObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]
    template['q'] = 2
    controlObsIDs.append(mw.runTrial('LLS.LLStoPuck', template, varList, dataID, 'dataAccel'))

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
    #samplePhaseData = random.uniform(0,1)
    #samplePhaseData = 0.0
    bestPhaseIndex = matchPhase(massObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(massObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(massObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]
    template['q'] = 2
    massObsIDs.append(mw.runTrial('LLS.LLStoPuck', template, varList, dataID, 'dataAccel'))

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
    #samplePhaseData = random.uniform(0,1)
    #samplePhaseData = 0.0
    bestPhaseIndex = matchPhase(inertiaObsID, samplePhaseData)
    obsState = mwOrbit.findObsState(inertiaObsID, bestPhaseIndex, ['dtheta','omega','q','v','delta'])
    for variable in obsState:
        template[variable] = obsState[variable]
    f = mwOrbit.findFootLocation(inertiaObsID, bestPhaseIndex, template['x'], template['y'], template['theta'])
    template['fx'] = f[0]
    template['fy'] = f[1]
    template['q'] = 2
    inertiaObsIDs.append(mw.runTrial('LLS.LLStoPuck', template, varList, dataID, 'dataAccel'))


print 'controlObsIDs: ' + str(controlObsIDs)
print 'massObsIDs: ' + str(massObsIDs)
print 'inertiaObsIDs: ' + str(inertiaObsIDs)


mw.saveTables()
