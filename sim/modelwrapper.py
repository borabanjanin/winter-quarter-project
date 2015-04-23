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
import scipy.optimize as spo
import lls as lls
import puck as puck
from opt import opt
#import modelplot as mp
from util import num
'''
t # time samples
a # cart accel
# set this member variable when you "uptake" experimental data
model.accel = ( lambda s,_,_ : np.array([0.,num.interp1(s,t,a),0.]) )
# later in LLS.dyn the following will be called:
LLSmdl.accel(.5,_,_)
'''
import os
import time
import sys
import json
import copy
import gc
import math
from collections import OrderedDict

class ModelWrapper(object):
    """
    Used to run and store simulation results

    Usage:

    """

    """
    Global Constants
    """

    def __init__(self, saveDir=None):
        self.saveDir = ModelWrapper.saveDirectory(saveDir)
        self.ptrialID = 0
        self.observations = {}
        self.parameters = {}
        self.configuration = {}
        self.trials = pd.DataFrame(columns=('ModelName','ConfFile','Cost','DataID','HashParm'))
        self.loadTables()
        self.loadCurrentID()
        self.data = {}

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

    def checkParmLengthNpy(self, dictionary=None):
        '''
        .checkParmLengthNpy finds the smallest list length in the
        dictionary data.

        INPUTS:
            data - m x n dictionary of numpy arrays

        OUTPUTS:
            int
        '''
        length = sys.maxint

        for parm in dictionary.keys():
            if len(np.ravel(dictionary[parm]))  < length:
                length = len(np.ravel(dictionary[parm]))

        if(length == sys.maxint):
            length = 0

        return length

    def checkParmLength(self, dictionary=None):
        '''
        .checkParmLength finds the smallest list length in the
        dictionary data.

        INPUTS:
            data - m x n dictionary of lists

        OUTPUTS:
            int
        '''
        length = sys.maxint

        for parm in dictionary.keys():
            if len(dictionary[parm])  < length:
                length = len(dictionary[parm])

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
        dictionary = {}

        for parm in listParm:
            if parm not in o.keys():
                raise Exception("ModelWrapper: Not valid list parameter")
            m =  eval("o" + "." + parm)
            dictionary[parm] = [np.ravel(m)]

        return dictionary

    def createDataFrameNpy(self, dataSize=0, dictionary=None):
        '''
        .createDataFrame takes the dictionary, data with arrays in the numpy format,
        and put the observations into a pandas dataframe.

        INPUTS:
            dataSize - 1 x 1 int
            data - m x n dictionary of lists

        OUTPUTS:
            dataframe - pandas object
        '''
        results = pd.DataFrame(columns=dictionary.keys(),index=range(dataSize))

        #print dictionary['acc'][0][1::3]

        for parm in dictionary.keys():
            results[parm] = np.ravel(dictionary[parm])

        return results

    def createDataFrame(self, dataSize=0, dictionary=None):
        '''
        .createDataFrame puts a dictionary of lists into a pandas dataframe.

        INPUTS:
            dataSize - 1 x 1 int
            data - m x n dictionary of lists

        OUTPUTS:
            dataframe - pandas object
        '''
        results = pd.DataFrame(columns=dictionary.keys(),index=range(dataSize))

        for parm in dictionary.keys():
            results[parm] = dictionary[parm]

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
        dictionary = self.unpackParm(o,listParm)
        dataSize = self.checkParmLengthNpy(dictionary)
        ob = self.createDataFrameNpy(dataSize,dictionary)
        self.observations[ID] = ob
        #TO DO: Fix model theta
        for i in range(len(self.observations[ID]['theta'])):
            self.observations[ID]['theta'][i] -= np.pi/2

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
        if isinstance(configName, str):
            self.trials.loc[ID] = [modelName, configName, np.nan, np.nan, parmHash]
        else:
            self.trials.loc[ID] = [modelName, np.nan, np.nan, np.nan, parmHash]
        #'ModelName','ConfFile','Cost','HashParm'

    def jsonConfiguration(self,configName):
        configFile = os.path.join(self.saveDir, 'configurations',configName)
        if os.path.isfile(configFile):
            with open(configFile, 'r') as file:
                return json.load(file)
        else:
            raise Exception("ModelWrapper: Not valid configuration file")

    def setAccel(self, model, accel, timeData, accelData):
        if accel == None or accel == 'noAccel':
            model.accel = ( lambda s,i,j : np.array([0.,0.,0.]))
        elif accel == 'dataAccel':
            model.accel = ( lambda s,i,j : np.array([0.,-num.interp1(np.array([s]),timeData,accelData),0.]))
        else:
            raise Exception("ModelWrapper: Not valid acceleration type")

    def initModel(self, modelName, config, dataID, accel):

        #configPath = os.path.join(os.getcwd(),config)

        #if os.path.isfile(configPath) == False:
        #    raise Exception("ModelWrapper: Not a valid config file")
        model = None

        if isinstance(config, dict):
            config = copy.copy(config)
            model = eval(modelName)(ModelConfiguration.packConfiguration(config))
        else:
            config = os.path.join(self.saveDir,'configurations',config)
            model = eval(modelName)(ModelConfiguration.packConfiguration(self.jsonConfiguration(config)))

        dt = model.p['dt']
        beginIndex, endIndex = self.findDataIndex(dataID, None, 'all')
        accelData = np.array(self.data[dataID].ix[beginIndex:endIndex-2, "CartAcceleration"])
        timeData = np.array(np.linspace(beginIndex*dt,(endIndex-2)*dt,endIndex-1-beginIndex))
        self.setAccel(model,accel,timeData,accelData)

        #mw.data[dataID].ix[beginIndex:endIndex-2, "CartAcceleration"]
        #model.accel = ( lambda s,i,j : np.array([0.,num.interp1(s,t,self.data[dataID]),0.]) )
        return model

    def runModel(self, model, listParm, confName):
        dbg  = model.p.get('dbg',False)
        #TO DO: Call simulate
        t,x,q = model(model.p['to'], model.p['tf'], model.x0, model.q0, model.p['N'], dbg)
        o = model.obs().resample(model.p['dt'])
        #o = model.obs()
        ID = self.newTrialID()
        self.storeObs(ID, o, listParm)
        self.storeParms(ID,listParm)
        self.storeConf(ID,model.p)
        self.storeMod(ID, model.name, confName)
        return ID

    def runTrial(self, modelName=None, config=None, listParm=None, dataID=None, accel=None):
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

        model = self.initModel(modelName, config, dataID,accel)
        ID = self.runModel(model, listParm, config)
        self.updateDataID(ID,dataID)
        return ID
        return 0

    def csvTrials(self):
        self.trials.to_csv(os.path.join(self.saveDir,'trials.csv'), sep='\t')

    def csvObservations(self):
        IDs = []
        for ID in self.observations:
            IDs.append(ID)
            self.csvObservation(ID)

        for ID in IDs:
            self.observations.pop(ID, None)

    def csvObservation(self,ID):
        observationsDF = self.createDataFrame(self.checkParmLength(self.observations[ID]), self.observations[ID])
        obsDir = os.path.join(self.saveDir,'observations')
        observationsDF.to_csv(os.path.join(obsDir,'observation-' + str(ID) + '.csv'), sep='\t')

    def saveTables(self):
        self.csvTrials()
        self.csvObservations()
        #self.jsonConfigurations()
        #self.jsonParameters()

    def csvLoadTrials(self):
        trialsDoc = os.path.join(self.saveDir,'trials.csv')
        if os.path.isfile(trialsDoc) == True:
            self.trials = pd.read_csv(trialsDoc, sep='\t', index_col=0)
            return True
        else:
            return False

    def checkFile(self, path):
        return os.path.isfile(path)

    def csvLoadObs(self, IDs):
        for ID in IDs:
            self.csvLoadOb(ID)

    def csvLoadOb(self,ID):
        savePath = os.path.join(self.saveDir,'observations')
        csvPath = os.path.join(savePath,'observation-' + str(ID) + '.csv')
        self.observations[ID] = pd.read_csv(csvPath, sep='\t', index_col=0)

    def loadTables(self):
        self.csvLoadTrials()
        #self.jsonLoadParameters()
        #self.jsonLoadConfigurations()

    def loadCurrentID(self):
        if len(self.trials.index) > 0:
            trialIDs = self.getTrialIDs()
            self.assignTrialID(max(trialIDs, key=lambda x: x))
            self.incTrialID()

    def csvLoadData(self, IDs):
        for ID in IDs:
            dataPath = os.path.join(os.getcwd(),'data','data-' + str(ID) + '.csv')
            self.data[ID] = pd.read_csv(dataPath, sep='\t', index_col=0)
        return IDs

    def csvReleaseData(self, IDs):
        for ID in IDs:
            self.data.pop(ID, None)

    def csvReleaseObs(self, IDs):
        for ID in IDs:
            self.observations.pop(ID, None)

    def updateCost(self, ID, cost):
        self.trials.ix[ID,'Cost'] = cost

    def updateDataID(self, ID, dataID):
        self.trials.ix[ID,'DataID'] = dataID

    def findDataIndex(self, dataID, offset, optMode):
        currentData = self.data[dataID]
        if optMode == 'pre':
            beginIndex = currentData[ModelOptimize.label['x']].first_valid_index()
            endIndex = offset
        elif optMode == 'post':
            beginIndex = offset
            endIndex = currentData[ModelOptimize.label['x']].last_valid_index()
        elif optMode == 'all':
            beginIndex = currentData[ModelOptimize.label['x']].first_valid_index()
            endIndex = currentData[ModelOptimize.label['x']].last_valid_index()
        else:
            raise Exception("Model Wrapper: Optimization mode not correctly set")

        return beginIndex, endIndex

    def findObsIndex(self, obsID):
        beginIndex = self.observations[obsID]['x'].first_valid_index()
        endIndex = self.observations[obsID]['x'].last_valid_index()
        return beginIndex, endIndex

    @staticmethod
    def rotMat3(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    @staticmethod
    def rotMat2(theta):
        return np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    def findObsState(self, obsID, index, variables):
        state = {}
        for var in variables:
            state[var] = self.observations[obsID].ix[index, var]
        return state

    def compareTrajectory(self, index0, index1, obsID0, obsID1, tolerance):
        """
        .compareTrajectory - Used to compare two identical trajectories except for the initial x,y,theta

        INPUTS:
            Index0 - 1 x 2 int
            Index1 - 1 x 2 int
            obsID0 - 1 x 1 int
            obsID1 - 1 x 1 int

        OUTPUTS:
            boolean - TRUE indicates the trajectory is the same
        """
        if (index1[1] - index1[0]) != (index0[1] - index0[0]):
            raise Exception("Model Wrapper: Trajectories have different lengths")

        if index0[0] not in list(self.observations[obsID0].index) or index0[1] not in list(self.observations[obsID0].index):
            raise Exception("Model Wrapper: obsID0 does not have all indexes")

        if index1[0] not in list(self.observations[obsID1].index) or index1[1] not in list(self.observations[obsID1].index):
            raise Exception("Model Wrapper: obsID1 does not have all indexes")

        #state0 = self.findObsState(obsID0, index0[0],['v','delta','dtheta'])
        #state1 = self.findObsState(obsID1, index1[0],['v','delta','dtheta'])
        #point0Start = np.matrix([[state0['v']],[state0['delta']],['dtheta']])
        #point1Start = np.matrix([[state1['v']],[state1['delta']],['dtheta']])

        for (i0,i1) in zip(range(index0[0],index0[1]),range(index1[0],index1[1])):
            row0 = self.observations[obsID0].ix[i0]
            row1 = self.observations[obsID1].ix[i1]
            point0Current = np.matrix([[row0['v']],[row0['delta']],[row0['omega']]])
            point1Current = np.matrix([[row1['v']],[row1['delta']],[row1['omega']]])

            #errorMat = ModelWrapper.rotMat3(state0['theta']) * (point0Current - point0Start)  - ModelWrapper.rotMat3(state1['theta']) * (point1Current - point1Start)

            errorMat = point0Current - point1Current


            if any(np.abs(x) > tolerance for x in errorMat):
                print 'index0: ' + str(i0)
                print 'index1: ' + str(i1)
                return False

        return True

    def findPeriod(self, obsID):
        observation = self.observations[obsID]
        maxIndex = max(observation.index)
        finalFootState = int(observation['q'].ix[maxIndex])
        if finalFootState == 0:
            footState = 1
            indexes = observation.query('q == 1').index
        else:
            footState = 0
            indexes = observation.query('q == 0').index

        index = len(indexes) - 1
        currentIndex = len(indexes) - 1
        i = 0

        while indexes[currentIndex] + i == indexes[index]:
            currentIndex -= 1
            i += 1
        return indexes[index] - indexes[currentIndex] + 1

    def findSpeed(self, obsID):
        period = self.findPeriod(obsID)
        observation = self.observations[obsID]
        maxIndex = max(observation.index)
        distanceSum = 0.0
        for i in range(maxIndex - period+1,maxIndex):
            xDiff =  observation.ix[i+1, 'x']- observation.ix[i, 'x']
            yDiff =  observation.ix[i+1, 'y']- observation.ix[i, 'y']
            distanceSum += np.sqrt(xDiff**2 + yDiff**2)
        return distanceSum

    def findPhase(self, sampleIndex, obsID):
        observation = self.observations[obsID]
        period = self.findPeriod(obsID)
        footstate = int(observation['q'].ix[sampleIndex])
        currentIndex = sampleIndex

        if footstate == 0:
            while observation.ix[currentIndex, 'q'] == observation.ix[sampleIndex, 'q']:
                currentIndex -= 1
            return (sampleIndex - (currentIndex + 1.0))/period
        else:
            while observation.ix[currentIndex, 'q'] == observation.ix[sampleIndex, 'q']:
                currentIndex -= 1
            while observation.ix[currentIndex, 'q'] == 0:
                currentIndex -= 1
            return (sampleIndex - (currentIndex + 1.0))/period

    def findFootLocation(self, obsID, sampleIndex, x, y, theta):
        #print 'new x, y, theta: ' + str(x) + ' '+ str(y) + ' '+ str(theta)
        observation = self.observations[obsID]
        obsState = self.findObsState(obsID, sampleIndex, ['x','y','theta','fx','fy'])
        #print 'obs x, y, theta: ' + str(obsState['x']) + ' '+ str(obsState['y']) + ' '+ str(obsState['theta'])
        #print 'obs fx, fy: ' + str(obsState['fx']) + ' ' + str(obsState['fy'])
        fOrigin = np.matrix([[obsState['fx'] - obsState['x']] ,[obsState['fy'] - obsState['y']]])
        offset = np.matrix([[x],[y]])
        f = ModelWrapper.rotMat2(theta - obsState['theta']) * fOrigin + offset
        #print 'new fx, fy: ' + str(float(f[0][0])) + ' ' + str(float(f[1][0]))
        #print ' '
        return (float(f[0][0]),float(f[1][0]))

    def phaseDiffData(self, data, index0, index1):
        return data.ix[index1, 'Roach_xv_phase'] - data.ix[index0, 'Roach_xv_phase']

    def findPhaseDiffData(self, data, offset, numPeriods):
        index = offset - 1
        while(self.phaseDiffData(data, index, offset) < numPeriods * 2.0*np.pi):
            index -= 1
        return offset - index

    def findPeriodData(self, data, offset, dt, numPeriods):
        periodSamples = self.findPhaseDiffData(data, offset, numPeriods)
        return (periodSamples * dt)/numPeriods

    def findAverageSpeedData(self, data, offset, numPeriods):
        totalSpeed = 0.0
        periodSamples = self.findPhaseDiffData(data, offset, numPeriods)
        r = range(offset-periodSamples, offset+1)
        for i in r:
            if np.isnan(data.ix[i, 'Roach_v']) == False:
                totalSpeed += data.ix[i, 'Roach_v']
        return totalSpeed/len(r)


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

    @staticmethod
    def packConfiguration(configuration):
        zVarList = ['v','delta','dtheta']
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

    def jsonSaveConfiguration(self,ID):
        #Data Compatibility with model
        self.configurations[ID] = self.packConfiguration(self.configurations[ID])

        fileName = 'config-' + str(ID) + '.json'
        savePath = os.path.join(self.saveDir, 'configurations',fileName)
        with open(savePath, 'w') as file:
            json.dump(self.configurations[ID], file)
            self.configurations.pop(ID, None)
        return fileName

    def jsonLoadConfigurations(self,IDs):
        for ID in IDs:
            self.jsonLoadConfiguration(ID)

    def jsonLoadConfiguration(self,ID):
        configPath = os.path.join(self.saveDir, 'configurations','config-' + str(ID) + '.json')
        if os.path.isfile(configPath):
            with open(configPath, 'r') as file:
                self.configurations[ID] = json.load(file)
        else:
            raise Exception("ModelConfiguration: Not valid configuration file ID: " + str(ID))

    def jsonLoadTemplate(self, fileName = 'template'):
        configPath = os.path.join(os.path.dirname(self.saveDir),fileName + '.json')
        if os.path.isfile(configPath):
            with open(configPath, 'r') as file:
                self.template = json.load(file)
                return self.template
        else:
            raise Exception("ModelConfiguration: Not valid template file")

    def generateConfiguration(self, template):
        ID = self.newConfID()
        self.configurations[ID] = copy.copy(template)
        return self.jsonSaveConfiguration(ID)

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

    def setConfValuesData(self, index, dataID, conf, variables):
        for var in variables:
            conf[var] = self.modelwrapper.data[dataID].ix[index, ModelOptimize.label[var]]
        return conf

    def setConfValuesObs(self, index, obsID, conf, variables):
        #print self.modelwrapper.observations[obsID]
        for var in variables:
            conf[var] = self.modelwrapper.observations[obsID].ix[index, var]
        return self.packConfiguration(conf)

    def setRunTime(self, beginIndex, endIndex, conf):
        conf['to'] = beginIndex * conf['dt']
        conf['tf'] = endIndex * conf['dt']
        return conf

class ModelOptimize(object):

    class Optimizer(object):

        def __init__(self, dataIDs, indVarOpt, sharedVarOpt, kwargs):
            self.dataIDs = dataIDs
            self.indVarOpt = indVarOpt
            self.sharedVarOpt = sharedVarOpt
            self.templates = {}
            self.masterTemplate = {}
            self.currentTrial = {}
            self.varIdentifier = OrderedDict()
            self.beginIndex = {}
            self.endIndex = {}
            self.trialNum = 0
            self.bounds = None
            self.modelName = ''

            for var in kwargs:
                setattr(self, var, kwargs[var])

        def setTrial(self, modelID, dataID):
            self.currentTrial[dataID] = modelID

        def setTrials(self, modelID):
            for dataID in self.dataIDs:
                self.setTrial(modelID, dataID)

        def setTemplate(self, template, dataID):
            self.templates[dataID] = copy.copy(template)

        def setTemplates(self, template):
            self.masterTemplate = copy.copy(template)
            for dataID in self.dataIDs:
                self.setTemplate(template, dataID)

        def setIndex(self, dataID, data):
            if hasattr(self,'optMode'):
                if self.optMode == 'pre':
                    self.beginIndex[dataID] = data[ModelOptimize.label['x']].first_valid_index()
                    self.endIndex[dataID] = self.offset
                elif self.optMode == 'post':
                    self.beginIndex[dataID] = self.offset
                    self.endIndex[dataID] = data[ModelOptimize.label['x']].last_valid_index()
            else:
                raise Exception("Model Optimize: Optimizer: No optimization mode set")

        def setIndexes(self, data):
            for dataID in self.dataIDs:
                self.setIndex(dataID, data[dataID])

        def setInitialValue(self, dataID, data):
            template = self.templates[dataID]
            for var in self.uptakeValues:
                template[var] = data.ix[self.beginIndex[dataID], ModelOptimize.label[var]]

        def setInitialValues(self, data):
            for dataID in self.dataIDs:
                self.setInitialValue(dataID, data[dataID])

        def setRunTime(self):
            for dataID in self.dataIDs:
                self.templates[dataID]['t'] = (self.endIndex[dataID]-self.beginIndex[dataID]+1) * self.templates[dataID]['dt']

        def refreshOptimizer(self, modelID, template, data):
            self.trialNum = 0
            self.templates = {}
            self.currentTrial = {}
            self.setTrials(modelID)
            self.setTemplates(template)
            self.setIndexes(data)
            self.setInitialValues(data)
            self.setRunTime()

        def propagateSharedVar(self):
            for var in self.sharedVarOpt:
                for dataID in self.dataIDs:
                    self.templates[dataID][var] = self.masterTemplate[var]

        def updateBoundVar(self, var, x0Max, x0Min):
            if var in ModelOptimize.boundValues:
                boundValue = ModelOptimize.boundValues[var]
                boundMax = boundValue[1]
                boundMin = boundValue[0]
            else:
                boundMax = np.inf
                boundMin = -np.inf
            x0Max.append(boundMax)
            x0Min.append(boundMin)
        def returnDecVar(self):
            x0 = []
            x0Max = []
            x0Min = []
            self.varIdentifier = OrderedDict()

            for var in self.sharedVarOpt:
                x0.append(self.masterTemplate[var])
                self.varIdentifier[var] = self.masterTemplate[var]
                self.updateBoundVar(var, x0Max, x0Min)

            for var in self.indVarOpt:
                varList = []
                for dataID in self.dataIDs:
                    self.updateBoundVar(var, x0Max, x0Min)
                    x0.append(self.templates[dataID][var])
                    varList.append(self.templates[dataID][var])
                self.varIdentifier[var] = varList

            self.bounds = ModelOptimize.Bounds(x0Max,x0Min)


            return x0

        def updateDecVar(self, x0):
            print 'Iteration: ' + str(self.trialNum)
            self.trialNum += 1

            i = 0
            for var in self.varIdentifier:
                if var in self.sharedVarOpt:
                    self.masterTemplate[var] = x0[i]
                    i += 1
                elif var in self.indVarOpt:
                    for j in range(0,len(self.dataIDs)):
                        self.templates[self.dataIDs[j]][var] = x0[i]
                        i += 1
                else:
                    raise Exception("Model Optimize: Optimizer: var not found in shared or individual list")
            self.propagateSharedVar()

    class Bounds(object):
        def __init__(self, xmax, xmin):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    #Penalty for missing a data point per index
    MISSING_PENALTY = 100

    columnIDs = {
        0:"CartX",
        1:"CartVelocity",
        2:"CartAcceleration",
        3:"TarsusCart1_x",
        4:"TarsusCart2_x",
        5:"TarsusCart3_x",
        7:"TarsusCart5_x",
        6:"TarsusCart4_x",
        8:"TarsusCart6_x",
        9:"TarsusCart1_y",
        10:"TarsusCart2_y",
        11:"TarsusCart3_y",
        12:"TarsusCart4_y",
        13:"TarsusCart5_y",
        14:"TarsusCart6_y",
        15:"TarsusBody1_x",
        16:"TarsusBody2_x",
        17:"TarsusBody3_x",
        18:"TarsusBody4_x",
        19:"TarsusBody5_x",
        20:"TarsusBody6_x",
        21:"TarsusBody1_y",
        22:"TarsusBody2_y",
        23:"TarsusBody3_y",
        24:"TarsusBody4_y",
        25:"TarsusBody5_y",
        26:"TarsusBody6_y",
        27:"TarsusCombined_x",
        28:"TarsusCombined_dx",
        29:"C0",
        30:"C1",
        31:"TarsusBody1_vx",
        32:"TarsusBody2_vx",
        33:"TarsusBody3_vx",
        34:"TarsusBody4_vx",
        35:"TarsusBody5_vx",
        36:"TarsusBody6_vx",
        37:"TarsusBody1_vy",
        38:"TarsusBody2_vy",
        39:"TarsusBody3_vy",
        40:"TarsusBody4_vy",
        41:"TarsusBody5_vy",
        42:"TarsusBody6_vy",
        43:"Roach_x",
        44:"Roach_y",
        45:"Roach_vx",
        46:"Roach_vy",
        47:"Roach_pitch",
        48:"Roach_roll",
        49:"Roach_yaw",
        50:"Roach_theta",
        51:"Roach_dtheta",
        52:"Roach_v",
        53:"Roach_heading",
        54:"Roach_omega",
        55:"Roach_phaser_phase",
        56:"Roach_phaser_residual",
        57:"Roach_xv_phase",
        58:"Roach_xv_residual",
    }

    label = {
        'x':'Roach_x',
        'y':'Roach_y',
        'v':'Roach_v',
        'theta':'Roach_theta',
        'omega':'Roach_dtheta'}

    boundValues = {
        'x':(-10,10),
        'y':(-10,10),
        'theta':(-15,15),
        'v':(0,200),
        'eta0':(0,5),
        'm':(0,10),
        'I':(0,10),
        'k':(0,10000),
        'beta':(0,5),
        'delta':(-5,5),
        'omega':(-5,5)}

    parmList = ["t","theta","x","y","fx","fy","v","delta","omega"]

    #TO DO: move to functions
    indexValues = {'begin':0,'end':50,'offset':283}

    varCost=['x','y']

    def __init__(self, modelwrapper):
        self.modelwrapper = modelwrapper
        self.mc = ModelConfiguration(modelwrapper)
        #ms = mp.ModelStreamer()
        #self.stream1, self.stream2 = ms.createConnection()
        self.optimizationData = {}
        self.treatments = pd.read_pickle('treatments.pckl')
        self.varOpt = None
        #self.varOpt = ['x','y','v','theta','omega']

    def displayBestTrial(self):
        best = pd.Series(self.modelwrapper.trials['Cost'])
        best.reset_index()
        bestIndex = (best[best == min(best)].index)[0]
        bestOB = self.modelwrapper.observations[bestIndex]
        print 'Best trial: ' + str(bestIndex)
        self.stream1.write(dict(x=list(bestOB['x']), y=list(bestOB['y'])))

    def createTemplate(self,templateName):
        template = self.mc.jsonLoadTemplate(templateName)
        self.varOpt.refreshOptimizer(-1,template, self.modelwrapper.data)

    @staticmethod
    def checkModelIndex(observation, i):
        if any(observation.index == i):
            return True
        else:
            return False

    '''
    def checkDataIndex(self, i, variables):
        for variables in variables:
            if isnull(data.ix[i,variable]):
                raise Exception("SimulatedAnealing: Null data " + str(i) + variable)
            else:
                return True
    '''

    def updateModel(self, x0):
        self.varOpt.updateDecVar(x0)


        for dataID in self.varOpt.dataIDs:
            ID = self.modelwrapper.runTrial(self.modelName,self.mc.generateConfiguration(self.varOpt.templates[dataID]),ModelOptimize.parmList, dataID)
            self.varOpt.currentTrial[dataID] = ID
            self.modelwrapper.updateDataID(ID,dataID)
            #Arbitrary large value
            self.modelwrapper.updateCost(ID, 99999999.00)

    def updateModelCosts(self, f):
        for dataID in self.varOpt.dataIDs:
            self.modelwrapper.updateCost(self.varOpt.currentTrial[dataID], f)

    @staticmethod
    def exponentialDecay(x, offset): return 100*math.exp(math.log(.5)/50*(offset-x))

    def trajectoryCost(self):
        summation = 0.0

        for dataID in self.varOpt.dataIDs:
            observation = self.modelwrapper.observations[self.varOpt.currentTrial[dataID]]
            for i in range(self.varOpt.beginIndex[dataID],self.varOpt.endIndex[dataID]+1):
                if ModelOptimize.checkModelIndex(observation,i-self.varOpt.beginIndex[dataID]):
                    for var in ModelOptimize.varCost:
                        dataName = ModelOptimize.label[var]
                        summation += (float(self.optimizationData[dataID].ix[i, dataName]) - float(observation.ix[i - self.varOpt.beginIndex[dataID], var]))**2 \
                        * ModelOptimize.exponentialDecay(ModelOptimize.indexValues['offset'],i)
                        if np.isnan(summation):
                            raise Exception("ModeOptimize: not valid row missing data: " + str(dataID) + ' i: ' + str(i + ModelOptimize.indexValues['offset']))
                else:
                    print 'WARNING: MODEL CRASHED'
                    summation += ModelOptimize.MISSING_PENALTY

        return summation

    def cost(self, x0):
        self.updateModel(x0)
        f = self.trajectoryCost()
        self.updateModelCosts(f)
        return f

    #The minimizer just returns values
    def noChange(fun, x0, args, **options):
        return spo.OptimizeResult(x=x0, fun=fun(x0), success=True, nfev=1)

    def gradientStep(self, currentCost, bestCost):
        return ((bestCost - currentCost) / (self.varOpt.s * bestCost)) * self.varOpt.stepSize

    def finiteDifferences(self, fun, x0, args, **options):
        f = self.cost(x0)
        print 'x0: ' + str(x0)
        print 'f: ' + str(f)
        xOpt = copy.copy(x0)
        fj = f

        for j in range(self.varOpt.gradientIterations):
            x1 = copy.copy(xOpt)
            for i in range(len(x0)):
                x2 = copy.copy(xOpt)
                x2[i] =  x1[i] + self.varOpt.s
                f2 = self.cost(x2)
                #x1[i] = ((fj - f2) / (self.varOpt.s * fj)) * self.varOpt.stepSize + x1[i]
                x1[i] = self.gradientStep(f2, fj) + x1[i]
            f1 = self.cost(x1)
            if f1 < fj:
                print 'Minimized'
                fj = f1
                xOpt = copy.copy(x1)
            else:
                break
        fOpt = fj

        '''
        for i in range(len(x0)):
            x1 = copy.copy(x0)
            x1[i] =  x0[i] + self.varOpt.s
            fi = self.cost(x1)
            xOpt[i] = ((fj - fi) / (self.varOpt.s * fj)) * self.varOpt.stepSize + x0[i]
        fOpt = self.cost(xOpt)
        '''

        print 'xOpt: ' + str(xOpt)
        print 'fOpt: ' + str(fOpt)
        if fOpt < f:
            return spo.OptimizeResult(x=xOpt, fun=fOpt, success=True, nfev=self.varOpt.gradientIterations)
        else:
            return spo.OptimizeResult(x=x0, fun=f, success=True, nfev=self.varOpt.gradientIterations)

    #Used to stream data to plotly
    #TO DO: handle modle optmization plots
    def streamTrials(self,ID):
        observation = self.modelwrapper.observations[ID]
        k=[]
        j=[]
        h=[]
        l=[]
        for i in range(ModelOptimize.indexValues['begin'],ModelOptimize.indexValues['end']+1):
            if self.checkModelIndex(observation,i):
                    k.append(observation.ix[i, 'x'])
                    j.append(observation.ix[i, 'y'])
                    #h.append(self.optimizationData[key].ix[i + ModelOptimize.indexValues['offset'], self.label['x']])
                    #l.append(self.optimizationData[key].ix[i + ModelOptimize.indexValues['offset'], self.label['y']])

        for key in self.optimizationData.keys():
            for i in range(ModelOptimize.indexValues['begin'],ModelOptimize.indexValues['end']+1):
                h.append(self.optimizationData[key].ix[i + ModelOptimize.indexValues['offset'], self.label['x']])
                l.append(self.optimizationData[key].ix[i + ModelOptimize.indexValues['offset'], self.label['y']])

        self.stream1.write(dict(x=k, y=j))
        self.stream2.write(dict(x=h, y=l))

    def optimizationLoop(self, x0):
        minimizer_kwargs = {"method":self.finiteDifferences, "jac":False}
        try:
            ret = spo.basinhopping(self.cost,x0,minimizer_kwargs=minimizer_kwargs, accept_test=self.varOpt.bounds, \
            niter=self.varOpt.iterations,T=10.,stepsize=5.,interval=1)
            #print ret
        except KeyboardInterrupt:
            print "exit optimization"

    def setOptimizationData(self, dataIDs):
        for dataID in dataIDs:
            self.optimizationData[dataID] = self.modelwrapper.data[dataID]

    def runOptimize(self, modelName, template, indVarOpt, sharedVarOpt, dataIDs, **kwargs):
        self.modelwrapper.csvLoadData(dataIDs)
        self.modelName = modelName
        self.varOpt = ModelOptimize.Optimizer(dataIDs, indVarOpt, sharedVarOpt, kwargs)
        self.createTemplate(template)
        self.setOptimizationData(dataIDs)
        self.optimizationLoop(self.varOpt.returnDecVar())
        #self.modelwrapper.csvReleaseData(dataIDs)
        #self.optimizationData = {}

    def findDataIDs(self, animalID = None, treatmentType = None):
        if animalID == None:
            return list(self.treatments.query('Treatment == "' + treatmentType + '"').index)
        elif treatmentType == None:
            return list(self.treatments.query('AnimalID == ' + str(animalID)).index)
        else:
            return list(self.treatments.query('Treatment == "' + treatmentType + '" and AnimalID == ' + str(animalID)).index)

    def runOptimizeAnimal(self, modelName, template, animalID, treatmentTypes, indVarOpt, sharedVarOpt, **kwargs):
        dataIDs = []
        for treatmentType in treatmentTypes:
            dataIDs += self.findDataIDs(animalID, treatmentType)
        print 'dataIDs: ' + str(dataIDs)
        self.runOptimize(modelName, template, indVarOpt, sharedVarOpt, dataIDs, **kwargs)

if __name__ == "__main__":
    mw = ModelWrapper()
    mo = ModelOptimize(mw)
    mc2 = ModelConfiguration(mw)

    kwargs = {'offset':283,'optMode':'pre', 'uptakeValues':['x','y','theta'], 'iterations': 1, 'stepSize': 0.01, 's': 0.0000001, 'gradientIterations':10}

    mo.runOptimize("lls.LLStoPuck","template",['x','y','theta'],[],[0],**kwargs)
    mw.saveTables()
