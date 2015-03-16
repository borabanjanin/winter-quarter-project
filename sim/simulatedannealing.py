import modelwrapper as mp
import matplotlib.pyplot as plt
import os
import time


saveDir = 'Test2'
currentDir = os.path.join(os.getcwd(),saveDir)
mw = mp.ModelWrapper(saveDir)

if not os.path.exists(os.path.join(currentDir,'pre')): os.makedirs(os.path.join(currentDir,'pre'))
if not os.path.exists(os.path.join(currentDir,'post')): os.makedirs(os.path.join(currentDir,'post'))

mo = mp.ModelOptimize(mw)
mc = mp.ModelConfiguration(mw)

# experimental trial #'s
#dataIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50]
#dataIDs = [0,1,2,3,4,5,6,7,8]
dataIDs = [0]

# optimization configuration
kwargs = {'offset':283,'optMode':'pre', 'uptakeValues':['x','y','theta'], 'iterations': 10, 'stepSize': 0.01, 's': 0.0001, 'gradientIterations':20}

mw.csvLoadData(dataIDs)


startTime = time.time()

# fit model to pre-perturbation data trial-by-trial
#mo.runOptimizeAnimal("lls.LLS", "template", None, ['control'], ['x','y','theta','v','eta0','beta','delta','omega','m','I','k'], [], **kwargs)

for dataID in dataIDs:
    mo.runOptimize("lls.LLS","template",['x','y','theta','v'],[],[dataID],**kwargs)
    mw.saveTables()

endTime = time.time()

# since we're using a stochastic algorithm, now find the best fit config for each trial
bestTrials = []
for dataID in dataIDs:
    dataIDData = mw.trials.query('DataID == ' + str(dataID))['Cost'].idxmin(1)
    bestTrials.append(dataIDData)

# load best fit obs's & config's
mw.csvLoadObs(bestTrials)
mc.jsonLoadConfigurations(bestTrials)

# "pack" optimized variables in config and update using experimental data
# TODO: use final condition from "best fit" config instead of experimental data
for i in range(len(dataIDs)):
    #Use data method
    #TODO: self.x0, self.q0 = self.extrinsic(self.p['z0'], self.p['q0'], self.p['x'], self.p['y'], self.p['theta'])
    #beginIndex, endIndex = mw.findDataIndex(dataIDs[i], kwargs['offset'], 'pre')
    #mc.setConfValuesData(kwargs['offset'], dataIDs[i], bestTrials[i], ['x','y','theta'])
    #mc.packConfiguration(mc.configurations[bestTrials[i]])
    #mc.setRunTime(beginIndex, endIndex, bestTrials[i])

    #Use obs method
    beginIndex, endIndex = mw.findObsIndex(bestTrials[i])
    mc.setConfValuesObs(endIndex - beginIndex, bestTrials[i], bestTrials[i], ['x','y','theta'])
    mc.packConfiguration(mc.configurations[bestTrials[i]])
    mc.setRunTime(beginIndex, endIndex, bestTrials[i])

# simulate forward through perturbation
pertModel = []
for confID in bestTrials:
    pertModel.append(mw.runTrial("lls.LLStoPuck",mc.configurations[confID],mp.ModelOptimize.parmList, dataID))

# plot
for i in range(len(dataIDs)):
    beginIndex, endIndex = mw.findDataIndex(dataIDs[i], kwargs['offset'], 'pre')
    dataX = mw.data[dataIDs[i]][mo.label['x']][beginIndex:endIndex+1]
    dataY = mw.data[dataIDs[i]][mo.label['y']][beginIndex:endIndex+1]
    obsX = mw.observations[bestTrials[i]]['x'][0:endIndex+1-beginIndex]
    obsY = mw.observations[bestTrials[i]]['y'][0:endIndex+1-beginIndex]
    plt.plot(dataX,dataY,'r--', label='Experimental Data')
    plt.plot(obsX,obsY,'b--', label='Model Observations')
    plt.legend(bbox_to_anchor=(1, 1),prop={'size':12},bbox_transform=plt.gcf().transFigure)
    plt.show()
    plt.savefig(os.path.join(mw.saveDir,'pre/pre-'+str(dataIDs[i])+".png"))
    plt.clf()

for i in range(len(dataIDs)):
    beginIndex, endIndex = mw.findDataIndex(dataIDs[i], kwargs['offset'], 'post')
    dataX = mw.data[dataIDs[i]][mo.label['x']][beginIndex:endIndex+1]
    dataY = mw.data[dataIDs[i]][mo.label['y']][beginIndex:endIndex+1]
    obsX = mw.observations[pertModel[i]]['x'][0:endIndex+1-beginIndex]
    obsY = mw.observations[pertModel[i]]['y'][0:endIndex+1-beginIndex]
    plt.plot(dataX,dataY,'r--', label='Experimental Data')
    plt.plot(obsX,obsY,'b--', label='Model Observations')
    plt.legend(bbox_to_anchor=(1, 1),prop={'size':12},bbox_transform=plt.gcf().transFigure)
    plt.show()
    plt.savefig(os.path.join(mw.saveDir,'post/post-'+str(dataIDs[i])+".png"))
    plt.clf()

print 'Total Time Elapsed: ' + str(endTime - startTime)

# TODO: save observations ?
#mw.saveTables()
