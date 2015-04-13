import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy



#saveDir = 'BestTrials-1k'

#index0 and index1 are tuples

varList = ['x','y','theta','fx','fy','hx','hy','E','dtheta','omega','PE','KE','q','v','delta']
#varList = ['t','x','y','theta','dx','dy','fx','fy','dtheta','delta','v','q']

mw = model.ModelWrapper()

mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

dataID = 1
rot0 = np.pi/4
rot1 = np.pi/3

mw.csvLoadData([dataID])

template = mc.jsonLoadTemplate("template")
template['dt'] = .002
template1 = copy.copy(template)

template['x'] = 2.0
template['y'] = -3.0
template['theta'] = rot0
template['t'] = 2

obsID0 = mw.runTrial('lls.LLS', template, varList, dataID)
index0 = mw.findObsIndex(obsID0)
state0 = mw.findObsState(obsID0, index0, varList)


sampleIndex = 40
template1 = mc.setConfValuesObs(sampleIndex, obsID0, template, varList)

template1['x'] = 4.0
template1['y'] = 5.0
template1['theta'] = rot1
template1['t'] = .5

f = mw.findFootLocation(obsID0, sampleIndex, template1['x'], template1['y'], rot1)

template1['fx'] = f[0]
template1['fy'] = f[1]

obsID1 = mw.runTrial('lls.LLS', template, varList, 1)
index1 = mw.findObsIndex(obsID1)

print index0
print index1

#setConfValuesObs(index, obsID, conf, variables)


print mw.compareTrajectory((sampleIndex,sampleIndex+10), (0,10), obsID0, obsID1, dataID)

plt.plot(mw.observations[obsID0]['x'],mw.observations[obsID0]['y'])
plt.plot(mw.observations[obsID1]['x'],mw.observations[obsID1]['y'])

#mw.saveTables()
'''

def R01(x) : return np.matrix([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])

firstRow0 = mw.observations[obsID0].ix[index0[0]]
firstRow1 = mw.observations[obsID1].ix[index1[0]]
lastRow0 = mw.observations[obsID0].ix[index0[1]]
lastRow1 = mw.observations[obsID1].ix[index1[1]]

point0S = np.matrix([[firstRow0['x']],[firstRow0['y']],[0.0]])
point1S = np.matrix([[firstRow1['x']],[firstRow1['y']],[0.0]])

point0E = np.matrix([[lastRow0['x']],[lastRow0['y']],[0.0]])
point1E = np.matrix([[lastRow1['x']],[lastRow1['y']],[0.0]])

#R01 = np.matrix([[np.cos(rot1), -np.sin(rot1), 0], [np.sin(rot1), np.cos(rot1), 0], [0, 0, 1]])

print R01(rot0) * (point0E - point0S)
print R01(rot1) * (point1E - point1S)
'''
