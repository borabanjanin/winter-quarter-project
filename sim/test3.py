import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy
import random



saveDir = 'StableOrbit'

#index0 and index1 are tuples

varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t']
#varList = ['t','x','y','theta','dx','dy','fx','fy','dtheta','delta','v','q']

mw = model.ModelWrapper(saveDir)

mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

dataID = 1
rot0 = np.pi/4
rot1 = np.pi/3

mw.csvLoadData([dataID])
mw.csvLoadObs([0])

template = mc.jsonLoadTemplate("templateControl")

#template['v'] = 35.5
#template['beta'] = 1.0
#template['eta0'] = 1.0

template['dt'] = .002
template['to'] = 1.0
template['tf'] = 5.0
template['N'] = 250
template1 = copy.copy(template)

template['x'] = 2.0
template['y'] = -3.0

'''
obsID0 = mw.runTrial('lls.LLS', template, varList, dataID)
index0 = mw.findObsIndex(obsID0)
state0 = mw.findObsState(obsID0, index0, varList)
'''

dataIDs = mo.treatments.query("Treatment == 'control'").index

mw.csvLoadData(dataIDs)

obsID0 = 0

random.seed(2)
for dataID in dataIDs:
    #for i in [20,520,1020,1520,2020,2200]:
    sampleIndex = 500
    template1 = mc.setConfValuesObs(sampleIndex, obsID0, template, ['dtheta','omega','q','v','delta'])

    template1['x'] = random.uniform(-10,10)
    template1['y'] = random.uniform(-10,10)
    template1['theta'] = random.uniform(0 , 2 * np.pi)

    f = mw.findFootLocation(obsID0, sampleIndex, template1['x'], template1['y'], template1['theta'])
    template1['fx'] = f[0]
    template1['fy'] = f[1]
        #plt.show()

    obsID1 = mw.runTrial('lls.LLStoPuck', template, varList, dataID)
    index1 = mw.findObsIndex(obsID1)

    #plt.figure(1)
    #plt.plot(mw.observations[obsID0]['t'],mw.observations[obsID0]['v'])
    #plt.plot(mw.observations[obsID1]['t'],mw.observations[obsID1]['v'])

    #plt.figure(2)
    #plt.plot(mw.observations[obsID0]['x'],mw.observations[obsID0]['y'])
    #plt.plot(mw.observations[obsID1]['x'],mw.observations[obsID1]['y'])
    #plt.show()

    #Trajectories don't match for larger sample indexs
    if len(mw.observations[obsID1].index) != 2000:
        print 'Shorter Index!!!'
        print 'DataID: ' + str(dataID)
        print 'sampleIndex: ' + str(sampleIndex)
        print
    elif mw.compareTrajectory((sampleIndex,sampleIndex+100), (0,100), obsID0, obsID1, .1) == False:
        print 'DataID: ' + str(dataID)
        print 'sampleIndex: ' + str(sampledIndex)
        print

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
