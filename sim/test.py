import modelwrapper as mp
import matplotlib.pyplot as plt
import os

mw = mp.ModelWrapper('TEST2')
mo = mp.ModelOptimize(mw)
mc = mp.ModelConfiguration(mw)

dataIDs = [0,1,2]

kwargs = {'offset':283,'optMode':'pre', 'uptakeValues':['x','y','theta','v'], 'iterations':10}

#mw.csvLoadData(dataIDs)

for dataID in dataIDs:
    mo.runOptimize("lls.LLStoPuck","template",['x','y', 'theta'],[],[dataID],**kwargs)
    #mw.saveTables()

bestTrials = []
for dataID in dataIDs:
    dataIDData = mw.trials.query('DataID == ' + str(dataID))['Cost'].idxmin(1)
    bestTrials.append(dataIDData)

mw.csvLoadObs(bestTrials)
mc.jsonLoadConfigurations(bestTrials)

for i in range(len(dataIDs)):
    mc.setConfValues(kwargs['offset'], dataIDs[i], bestTrials[i], ['x','y','theta'])
    mc.packConfiguration(mc.configurations[bestTrials[i]])

pertModel = []
for confID in bestTrials:
    pertModel.append(mw.runTrial("lls.LLStoPuck",mc.configurations[confID],mp.ModelOptimize.parmList, dataID))

for i in range(len(dataIDs)):
    beginIndex, endIndex = mw.findDataIndex(dataIDs[i], kwargs['offset'], 'pre')
    dataX = mw.data[dataIDs[i]][mo.label['x']][beginIndex:endIndex+1]
    dataY = mw.data[dataIDs[i]][mo.label['y']][beginIndex:endIndex+1]
    obsX = mw.observations[bestTrials[i]]['x'][0:endIndex+1-beginIndex]
    obsY = mw.observations[bestTrials[i]]['y'][0:endIndex+1-beginIndex]
    plt.plot(dataX,dataY,'r--',obsX,obsY,'b--')
    plt.show()
    plt.savefig(os.path.join(mw.saveDir,'pre-'+str(i)+".png"))
    plt.clf()

for i in range(len(dataIDs)):

    beginIndex, endIndex = mw.findDataIndex(dataIDs[i], kwargs['offset'], 'post')
    dataX = mw.data[dataIDs[i]][mo.label['x']][beginIndex:endIndex+1]
    dataY = mw.data[dataIDs[i]][mo.label['y']][beginIndex:endIndex+1]
    obsX = mw.observations[pertModel[i]]['x'][0:endIndex+1-beginIndex]
    obsY = mw.observations[pertModel[i]]['y'][0:endIndex+1-beginIndex]
    plt.plot(dataX,dataY,'r--',obsX,obsY,'b--')
    plt.show()
    plt.savefig(os.path.join(mw.saveDir,'post-'+str(i)+".png"))
    plt.clf()

mw.saveTables()
