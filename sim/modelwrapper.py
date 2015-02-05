#!/usr/bin/python
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2014

import pandas as pd
import numpy as np

import lls as lls
import puck as puck
from opt import opt
import modelplot as mp

import os
import time
import sys
import json
import copy

class ModelWrapper(object):
    """
    Used to run and store simulation results

    Usage:

    """

    """
    Global Constants
    """
    NUMBER_MODELS = 1000

    def __init__(self, saveDir=None):
        self.saveDir = ModelWrapper.saveDirectory(saveDir)
        self.ptrialID = 0
        self.observations = {}
        self.parameters = {}
        self.configuration = {}
        self.trials = pd.DataFrame(columns=('ModelName','ConfFile','Cost','HashParm'))
        self.loadTables()
        self.loadCurrentID()

    @staticmethod
    def hashaList(inputList):
        '''
        .hashaList returns a hash of an inputted list.

        INPUTS:
            inputList - 1 x n list of elements

        OUTPUTS:
            int
        '''
        combinedString = ''

        for string in inputList:
            combinedString += str(string)

        return hash(combinedString)

    @staticmethod
    def saveDirectory(currentDate=None):

        if currentDate == None:
            currentDate = time.strftime("%Y%m%d")

        savePath = os.path.join(os.getcwd(),currentDate)

        if not os.path.exists(savePath):
            os.makedirs(savePath)
            os.makedirs(os.path.join(savePath,'observations'))
            os.makedirs(os.path.join(savePath,'configurations'))

        return savePath

    def trialID(self):
        '''
        .trialID returns the current trialID.

        OUTPUTS:
            int
        '''
        return self.ptrialID

    def assignTrialID(self, newID):
        if self.ptrialID >= sys.maxint:
            raise Exception("ModelWrapper: No new keys available")
        self.ptrialID = newID

    def incTrialID(self):
        '''
        .incTrialID increments the trialID.
        '''
        self.ptrialID += 1

        if self.ptrialID >= sys.maxint:
            raise Exception("ModelWrapper: No new keys available")

    def newTrialID(self):
        '''
        .newTrialID returns a new trialID

        OUTPUTS:
            int
        '''
        self.incTrialID()
        return self.trialID()-1

    def getTrialIDs(self):
        return self.trials.index.get_values()

    def printGaitStats(self, z=None):
        '''
        .printGaitStats prints the time to compute the gait.

        INPUTS:
            z - 1 x 1 the time spent finding the gait

        OUTPUTS:
            int
        '''
        st = time.time()
        print '%0.2f sec to find gait, z = %s' % (time.time() - st,z)

    def checkParmLengthNpy(self, data=None):
        '''
        .checkParmLengthNpy finds the smallest list length in the
        dictionary data.

        INPUTS:
            data - m x n dictionary of numpy arrays

        OUTPUTS:
            int
        '''
        length = sys.maxint

        for parm in data.keys():
            if len(np.ravel(data[parm]))  < length:
                length = len(np.ravel(data[parm]))

        if(length == sys.maxint):
            length = 0

        return length

    def checkParmLength(self, data=None):
        '''
        .checkParmLength finds the smallest list length in the
        dictionary data.

        INPUTS:
            data - m x n dictionary of lists

        OUTPUTS:
            int
        '''
        length = sys.maxint

        for parm in data.keys():
            if len(data[parm])  < length:
                length = len(data[parm])

        if length == sys.maxint:
            length = 0

        return length

    @staticmethod
    def dataLength(observation):
        return max(observation.index.get_values(), key=lambda x: x)

    @staticmethod
    def minDataLength(observations):
        if len(observations.keys > 0):
            minimum = int("inf")
            for ID in observations.keys():
                observation = observations[ID]
                length = ModelWrapper.dataLength(observation)
                if length < minimum:
                    minimum = length
            return minimum
        else:
            return 0

    def unpackParm(self, o, listParm):
        '''
        .unpackParm takes the dictionary, o,
        and puts the tracked parameters into another
        dictionary, data. Goes from numpy array format to a
        python list.

        INPUTS:
            listParm - 1 x n list
            o - k x n dictionary of lists

        OUTPUTS:
            data - m x n dictionary of lists
        '''
        data = {}

        for parm in listParm:
            if parm not in o.keys():
                raise Exception("ModelWrapper: Not valid list parameter")
            m =  eval("o" + "." + parm)
            data[parm] = [np.ravel(m)]

        return data

    def packConfiguration(self, configuration):
        zVarList = ['v','delta','omega']
        qVarList = ['q','m','I','eta0','k','d','beta','fx','fy']
        z0 = []
        q0 = []
        for var in zVarList:
            z0.append(configuration[var])
        for var in qVarList:
            q0.append(configuration[var])
        configuration['q0'] = q0
        configuration['z0'] = z0
        return configuration

    def createDataFrameNpy(self, dataSize=0, data=None):
        '''
        .createDataFrame takes the dictionary, data with arrays in the numpy format,
        and put the observations into a pandas dataframe.

        INPUTS:
            dataSize - 1 x 1 int
            data - m x n dictionary of lists

        OUTPUTS:
            dataframe - pandas object
        '''
        results = pd.DataFrame(columns=data.keys(),index=range(dataSize))

        for parm in data.keys():
            results[parm] = np.ravel(data[parm])

        return results

    def createDataFrame(self, dataSize=0, data=None):
        '''
        .createDataFrame puts a dictionary of lists into a pandas dataframe.

        INPUTS:
            dataSize - 1 x 1 int
            data - m x n dictionary of lists

        OUTPUTS:
            dataframe - pandas object
        '''
        results = pd.DataFrame(columns=data.keys(),index=range(dataSize))

        for parm in data.keys():
            results[parm] = data[parm]

        return results

    def storeObs(self, ID, o, listParm):
        '''
        .storeObs takes the trialID, optimization parameters,
        and tracked parameters and stores them in a dictionary. The
        dictionary has key values which correspond to the trialID and
        the values are pandas dataframes.

        INPUTS:
            ID - 1 x 1 int
            o - m x n dictionary of lists
            listParm - 1 x n list

        OUTPUTS:
            observations - m X n dictionary
        '''
        data = self.unpackParm(o,listParm)
        dataSize = self.checkParmLengthNpy(data)
        ob = self.createDataFrameNpy(dataSize,data)
        self.observations[ID] = ob

    def storeParms(self, ID, listParm):
        '''
        .storeParms takes the trialID and tracked parameters
        and stores them in a dictionary. The dictionary has key values
        which correspond to the trialID and the values are lists of
        parameters tracked for the specific run.

        INPUTS:
            ID - 1 x 1 int
            listParm - 1 x n list

        OUTPUTS:
            parameters - m x n dictionary
        '''

        self.parameters[ID] = copy.copy(listParm)

    def storeConf(self, ID, p):
        '''
        .storeConf takes the trialID and configuration parameters
        and stores them in a dictionary. The dictionary has key values
        which correspond to the trialID and the values are lists of
        configuration parameters for the specific run.

        INPUTS:
            ID - 1 x 1 int
            p - m x n dictionary

        OUTPUTS:
            parameters - m x n dictionary
        '''

        self.configuration[ID] = p.items()

    def storeMod(self, ID, modelName, configName):
        '''
        .storedMod takes the trialID and model call. Then it makes a hash of
        the respective configuration and tracked parameters. It then stores
        the model call, configuration hash and tracked hash in
        a pandas dataframe with the index corresponding to the trialID.

        INPUTS:
            ID - 1 x 1 int
            p - 1 x n list

        OUTPUTS:
            parameters - m x n dictionary
        '''
        parmHash = self.hashaList(self.parameters[ID])
        self.trials.loc[ID] = [modelName, parmHash, np.nan, configName]
        #'ModelName','ConfFile','Cost','HashParm'
    def jsonLoadConfiguration(self,configName):
        configFile = os.path.join(self.saveDir, 'configurations',configName)
        if os.path.isfile(configFile):
            with open(configFile, 'r') as file:
                return json.load(file)
        else:
            raise Exception("ModelWrapper: Not valid configuration file")

    def initModel(self, modelName, config):

        #configPath = os.path.join(os.getcwd(),config)

        #if os.path.isfile(configPath) == False:
        #    raise Exception("ModelWrapper: Not a valid config file")
        model = eval(modelName)(self.packConfiguration(self.jsonLoadConfiguration(config)))
        return model

    def runModel(self, model, listParm, confName):
        dbg  = model.p.get('dbg',False)
        #TO DO: Call simulate
        t,x,q = model(0, 1e99, model.x0, model.q0, model.p['N'], dbg)
        o = model.obs().resample(model.p['dt'])
        ID = self.newTrialID()
        self.storeObs(ID, o, listParm)
        self.storeParms(ID,listParm)
        self.storeConf(ID,model.p)
        self.storeMod(ID, model.name, confName)
        return ID

    def runTrial(self, modelName=None, config=None, listParm=None):
        '''
        .runTrial takes the model call, configuration file, and list of
        parameters to be tracked. It then runs the model and stores the
        observations, parameters tracked, and configurations for the respective
        trial.

        INPUTS:
            ID - 1 x 1 int
            p - 1 x n list

        OUTPUTS:
            parameters - m x n dictionary
        '''

        if modelName == None \
        or config == None \
        or listParm == None:
            raise Exception("ModelWrapper: No model name, config file, or list of variables to track was given")
        model = self.initModel(modelName, os.path.join(self.saveDir,'configurations',config))
        return self.runModel(model, listParm, config)

    def csvTrials(self):
        self.trials.to_csv(os.path.join(self.saveDir,'trials.csv'), sep='\t')

    def csvObservations(self):
        for ID in self.getTrialIDs():
            observationsDF = self.createDataFrame(self.checkParmLength(self.observations[ID]), self.observations[ID])
            obsDir = os.path.join(self.saveDir,'observations')
            observationsDF.to_csv(os.path.join(obsDir,'observation' + str(ID) + '.csv'), sep='\t')

    def saveTables(self):
        self.csvTrials()
        self.csvObservations()
        #self.jsonConfigurations()
        #self.jsonParameters()

    def __del__(self):
        self.saveTables()

    def csvLoadTrials(self):
        trialsDoc = os.path.join(self.saveDir,'trials.csv')
        if os.path.isfile(trialsDoc) == True:
            self.trials = pd.read_csv(trialsDoc, sep='\t', index_col=0)
            return True
        else:
            return False

    def checkFile(self, path):
        return os.path.isfile(path)

    def csvLoadObs(self):
        savePath = os.path.join(self.saveDir,'observations')
        for ID in self.getTrialIDs():
            csvPath = os.path.join(savePath,'observation' + str(ID) + '.csv')
            self.observations[ID] = pd.read_csv(csvPath, sep='\t', index_col=0)

    def loadTables(self):
        if self.csvLoadTrials() == True:
            self.csvLoadObs()
        #self.jsonLoadParameters()
        #self.jsonLoadConfigurations()

    def loadCurrentID(self):
        if len(self.trials.index) > 0:
            trialIDs = self.getTrialIDs()
            self.assignTrialID(max(trialIDs, key=lambda x: x))
            self.incTrialID()

    def updateCost(self, ID, cost):
        self.trials.ix[ID,'Cost'] = cost

class ModelConfiguration(object):

    def __init__(self, modelwrapper,template = None):
        self.template = template
        self.modelwrapper = modelwrapper
        self.pConfID = 0
        self.saveDir = modelwrapper.saveDir
        self.configurations = {}
        self.loadCurrentID()


    def loadCurrentID(self):
        IDs = self.loadConfIDs()
        if len(IDs) > 1:
            self.assignConfID(max(IDs, key=lambda x: x))
            self.incConfID()

    def confID(self):
        '''
        .trialID returns the current trialID.

        OUTPUTS:
            int
        '''
        return self.pConfID

    def assignConfID(self, newID):
        if self.pConfID >= sys.maxint:
            raise Exception("ModelConfiguration: No new keys available")
        self.pConfID = newID

    def resetConfID(self):
        self.pConfID = 0

    def incConfID(self):
        '''
        .incTrialID increments the trialID.
        '''
        #hold = self.pConfID
        #self.pConfID = hold + 1
        self.pConfID += 1
        if self.pConfID >= sys.maxint:
            raise Exception("ModelConfiguration: No new keys available")

    def newConfID(self):
        '''
        .newTrialID returns a new trialID

        OUTPUTS:
            int
        '''
        self.incConfID()
        return self.confID()-1

    def jsonSaveConfiguration(self,ID):
        fileName = 'config-' + str(ID) + '.json'
        savePath = os.path.join(self.saveDir, 'configurations',fileName)
        with open(savePath, 'w') as file:
            json.dump(self.configurations[ID], file)
            self.configurations.pop(ID, None)
        return fileName

    def jsonLoadConfiguration(self,ID):
        configPath = os.path.join(self.saveDir, 'configurations')
        if os.path.isfile(configPath):
            with open(os.path.join(configPath,'config' + str(ID) + '.json'), 'r') as file:
                self.configurations[ID] = json.load(file)
        else:
            raise Exception("ModelConfiguration: Not valid configuration file")

    def jsonLoadTemplate(self, fileName = 'template'):
        configPath = os.path.join(os.path.dirname(self.saveDir),fileName + '.json')
        if os.path.isfile(configPath):
            with open(configPath, 'r') as file:
                self.template = json.load(file)
                return self.template
        else:
            raise Exception("ModelConfiguration: Not valid configuration file")

    def generateConfiguration(self, var):
        ID = self.newConfID()
        for key in var.keys():
            self.template[key] = var[key]
        self.configurations[ID] = copy.copy(self.template)
        return ID

    def loadConfIDs(self):
        IDs = []
        savePath = os.path.join(self.saveDir, 'configurations')
        for fileName in os.listdir(savePath):
            IDs.append(int(fileName.split('-')[1].split('.')[0]))
        return IDs

    def loadConfNames(self):
        confNames = []
        savePath = os.path.join(self.saveDir, 'configurations')
        for fileName in os.listdir(savePath):
            confNames.append(fileName)
        return confNames


if __name__ == "__main__":
    mw = ModelWrapper()
    #mc = ModelConfiguration(mw)
    #mc.jsonLoadTemplate('template')
    Vars = {}
    Vars['d'] = -0.3
    #tprint mc.loadConfNames()
    #mc.jsonSaveConfiguration(mc.generateConfiguration(Vars))
    #mc.jsonSaveConfiguration(mc.generateConfiguration(Vars))

    #testKey = test.keys()
    #test = dict(test.pop("21", None))
    #print test
    #mc.jsonConfigurations(test)


    #print dict(test)

    mw.animateTrialID([0,1,3,2])
    #mw.animateTrialID([1])
    #mw.runTrial("lls.LLS",'config-1.json',["t","theta","x","y","fx","fy"])
    #mw.runTrial("lls.LLS","lls.cfg",["t","x","fx","fy"])
