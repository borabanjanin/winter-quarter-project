import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np

saveDir = 'CartAccel'
varList = ['x','y','theta']

dataIDs = range(5)

# MoI : control - 2.4538748499999996, inertia - 21.039901266666668 , mass - 2.4538748499999996
# Mass : control - 2.88, inertia - 3.7079999999999997, mass - 3.95

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

mw.csvLoadData(dataIDs)

sampleIndex = 283
template = mc.jsonLoadTemplate('template')

obsIDs = []

for dataID in dataIDs:
    template = mc.setConfValuesData(sampleIndex, dataID, template, varList)
    (beginIndex, endIndex) = mw.findDataIndex(dataID, sampleIndex, 'post')
    template = mc.setRunTime(beginIndex, endIndex, template)
    obsID = mw.runTrial('lls.LLS', template, varList, dataID, 'dataAccel')
    obsIDs.append(obsID)

mw.saveTables()

dataTable =  mp.combineData(dataIDs, sampleIndex)
obsTable = mp.combineObs(obsIDs , sampleIndex)

dataX =  dataTable.groupby('SampleID').mean()['Roach_x']
dataY =  dataTable.groupby('SampleID').mean()['Roach_y']
obsX =  obsTable.groupby('SampleID').mean()['x']
obsY =  obsTable.groupby('SampleID').mean()['y']
print len(obsX.index)
print len(dataX.index)
#c =  dataTable.groupby('SampleID').quantile(q=[.025,.975])['CartAcceleration']/(100*9.81)

plt.clf()
plt.figure(1)
plt.plot(dataY[0:], 'b', lw=4)
plt.plot(obsY, 'g', lw=4)
#plt.plot(b.index,c[:,0.025], 'k--')
#plt.plot(b.index,c[:,0.975], 'k--')
