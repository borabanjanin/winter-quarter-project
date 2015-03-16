import modelwrapper as model
import matplotlib.pyplot as plt
import os
import json
import time
import pandas as pd

uptakeDirectories = ['0-optFolder']
saveDir = 'Test'

bestTrials = pd.DataFrame(columns=('ModelName','ConfFile','Cost','DataID','HashParm'))
bestConfigurations = {}
bestObservations = {}

mw = model.ModelWrapper(uptakeDirectories[0])
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

def uptakeBestConfiguration(trials, trialID, dataID):
    confFile = trials.ix[trialID, 'ConfFile']
    with open(os.path.join(mw.saveDir,'configurations',confFile),'r') as file:
        bestConfigurations[dataID] = json.load(file)

    bestTrials.ix[int(dataID), 'ConfFile'] = 'config-' + str(int(dataID)) + '.json'

def uptakeBestObservations(trials, trialID, dataID):
    csvPath = os.path.join(mw.saveDir,'observations','observation-' + str(int(trialID)) + '.csv')
    bestObservations[dataID] = pd.read_csv(csvPath, sep='\t', index_col=0)

def reconcileBestTrial(trials, trialID, dataID):
    if dataID in list(bestTrials['DataID']):
        currentCost = float(bestTrials.query('DataID == ' + str(dataID))['Cost'])
        newCost = trials.loc[trialID]['Cost']
        if newCost < currentCost:
            index = bestTrials.query('DataID == ' + str(dataID)).index
            bestTrials.loc[dataID] = list(trials.loc[trialID])
            uptakeBestConfiguration(trials, trialID, dataID)
            uptakeBestObservations(trials, trialID, dataID)
    else:
        bestTrials.loc[dataID] = trials.loc[trialID]
        uptakeBestConfiguration(trials, trialID, dataID)
        uptakeBestObservations(trials, trialID, dataID)


def findBestTrial(trials):
    dataIDs = pd.unique(trials['DataID'])
    for i in range(len(dataIDs)):
        dataID = int(dataIDs[i])
        trialID = trials.query('DataID == ' + str(dataID))['Cost'].idxmin(1)
        print 'dataID: ' + str(dataID) + ' trialID: ' +str(trialID)
        reconcileBestTrial(trials, trialID, dataID)

for directory in uptakeDirectories:
    mw = model.ModelWrapper(directory)
    findBestTrial(mw.trials)


mw = model.ModelWrapper(saveDir)
mc = model.ModelConfiguration(mw)

mw.trials = bestTrials
mw.observations = bestObservations
mc.configurations = bestConfigurations

for confID in mc.configurations.keys():
    mc.jsonSaveConfiguration(confID)

mw.saveTables()
