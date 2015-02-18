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
import modelplot as mp

import os
import time
import sys
import json
import copy
import gc

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
        self.trials = pd.DataFrame(columns=('ModelName','ConfFile','Cost','DataIDs','HashParm'))
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
        self.trials.loc[ID] = [modelName, configName, np.nan, np.nan, parmHash]
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
        t,x,q = model(0, 3, model.x0, model.q0, model.p['N'], dbg)
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
        IDs = []
        for ID in self.observations:
            IDs.append(ID)
            self.csvObservation(ID)

        for ID in IDs:
            self.observations.pop(ID, None)

    def csvObservation(self,ID):
        observationsDF = self.createDataFrame(self.checkParmLength(self.observations[ID]), self.observations[ID])
        obsDir = os.path.join(self.saveDir,'observations')
        observationsDF.to_csv(os.path.join(obsDir,'observation' + str(ID) + '.csv'), sep='\t')

    def saveTables(self):
        self.csvTrials()
        #self.csvObservations()
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
        csvPath = os.path.join(savePath,'observation' + str(ID) + '.csv')
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

    def updateDataIDs(self, ID, dataIDs):
        stringIDs = ''
        for dataID in dataIDs:
            stringIDs += str(dataID) + ' '
        self.trials.ix[ID,'DataIDs'] = stringIDs

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

class ModelOptimize(object):

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
        24:"






        TarsusBody4_y",
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

    parmList = ["t","theta","x","y","fx","fy","v","delta","omega"]

    #TO DO: move to functions
    indexValues = {'begin':0,'end':200,'offset':283}

    varCost=['x','y']

    def __init__(self, modelwrapper):
        self.modelwrapper = modelwrapper
        self.mc = ModelConfiguration(modelwrapper)
        ms = mp.ModelStreamer()
        self.stream1, self.stream2 = ms.createConnection()
        self.optimizationData = {}
        self.dataIDs = []
        self.treatments = pd.read_pickle('treatments.pckl')
        #self.varOpt = ['x','y','v','theta','omega']

    def displayBestTrial(self):
        best = pd.Series(self.modelwrapper.trials['Cost'])
        best.reset_index()
        bestIndex = (best[best == min(best)].index)[0]
        bestOB = self.modelwrapper.observations[bestIndex]
        print 'Best trial: ' + str(bestIndex)
        self.stream1.write(dict(x=list(bestOB['x']), y=list(bestOB['y'])))

    def runTemplate(self,templateName):
        template = self.mc.jsonLoadTemplate(templateName)
        template = self.modelwrapper.packConfiguration(template)
        x0 = self.initialGuess(template)
        ID = self.modelwrapper.runTrial("lls.LLS",self.mc.jsonSaveConfiguration(self.mc.generateConfiguration(template)),ModelOptimize.parmList)
        return x0, ID

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

    def cost(self,x0):
        observation = self.modelwrapper.observations[self.modelwrapper.trialID()-1]
        summation = 0.0
        for i in range(ModelOptimize.indexValues['begin'],ModelOptimize.indexValues['end']+1):
            if ModelOptimize.checkModelIndex(observation,i):
                currentDiff = []
                for var in ModelOptimize.varCost:
                    dataName = ModelOptimize.label[var]
                    for dataID in self.optimizationData.keys():
                        summation += (float(self.optimizationData[dataID].ix[i + ModelOptimize.indexValues['offset'], dataName]) - float(observation.ix[i, var]))**2
                        if np.isnan(summation):
                            raise Exception("ModeOptimize: not valid row missing data: " + str(dataID) + ' i: ' + str(i))
            else:
                summation += ModelOptimize.MISSING_PENALTY
        return summation

    #The minimizer just returns values
    @staticmethod
    def noChange(fun, x0, args, **options):
        return spo.OptimizeResult(x=x0, fun=fun(x0), success=True, nfev=1)

    #Used to stream data to plotly
    #TO DO: handle modle optmization plots
    def streamTrial(self,ID):
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

    #Used in basinhopping callback
    #ModelWrapper hook
    def modelSimulate(self, x, f, record):
        #print f
        variables = {}
        for i in range(x.size):
            variables[self.varOpt[i]] = x[i]
        confID = self.mc.generateConfiguration(variables)
        confFile = self.mc.jsonSaveConfiguration(confID)
        ID = self.modelwrapper.runTrial("lls.LLS",confFile,ModelOptimize.parmList)
        if ID % 20 == 0:
            self.streamTrial(ID)
        self.modelwrapper.updateCost(ID-1, f)
        self.modelwrapper.updateDataIDs(ID,self.optimizationData.keys())
        #Arbitrary large value
        self.modelwrapper.updateCost(ID, 1000000.00)

    def initialGuess(self,template):
        x0 = []
        for var in self.varOpt:
            x0.append(template[var])
        return x0

    def optimizationLoop(self, x0):
        minimizer_kwargs = {"method":ModelOptimize.noChange, "jac":False}
        try:
            #self.setSPOVars()
            #cost_ = lambda x : self.cost(x)
            ret = spo.basinhopping(self.cost,x0,minimizer_kwargs=minimizer_kwargs, \
            niter=100,callback=self.modelSimulate,T=10.,stepsize=2.,interval=2)
            print ret
        except KeyboardInterrupt:
            print "exit optimization"

    def setOptimizationData(self, dataIDs):
        for dataID in dataIDs:
            self.optimizationData[dataID] = self.modelwrapper.data[dataID]

    def runOptimize(self, modelName, template, indVarOpt, sharedVarOpt, dataIDs):
        x0, configID = self.runTemplate(template)
        self.modelwrapper.csvLoadData(dataIDs)
        self.setOptimizationData(dataIDs)
        self.optimizationLoop(x0)
        self.modelwrapper.csvReleaseData(dataIDs)
        self.optimizationData = {}

    def findDataIDs(self, animalID = None, treatmentType = None):
        if animalID == None:
            return list(self.treatments.query('Treatment == "' + treatmentType + '"').index)
        elif treatmentType == None:
            return list(self.treatments.query('AnimalID == ' + str(animalID)).index)
        else:
            return list(self.treatments.query('Treatment == "' + treatmentType + '" and AnimalID == ' + str(animalID)).index)

    '''
    def runOptimizeAnimal(self, modelName, template, animalID, treatmentTypes):
        dataIDs = []
        for treatmentType in treatmentTypes:
            dataIDs += self.findDataIDs(animalID, treatmentType)
        print dataIDs
        self.runOptimize(modelName, template, dataIDs)
    '''

if __name__ == "__main__":
    mw = ModelWrapper()
    mo = ModelOptimize(mw)
    mc2 = ModelConfiguration(mw)

    mo.runOptimize("lls.LLStoPuck","template",['x','y','v'],['k'],[0,1,2])
    #mo.runOptimizeAnimal("lls.LLS","template",2,['control','mass'])
    #mo.displayBestTrial()
    #print mo.treatments

    mw.csvObservations()
    mw.saveTables()

    #anim = mp.ModelAnimate(mw)
    #anim.animateTrialID([0,1])

    #mc = ModelConfiguration(mw)
    #mc.jsonLoadTemplate('template')
    #Vars = {}
    #Vars['d'] = -0.3
    #tprint mc.loadConfNames()
    #mc.jsonSaveConfiguration(mc.generateConfiguration(Vars))
    #mc.jsonSaveConfiguration(mc.generateConfiguration(Vars))

    #testKey = test.keys()
    #test = dict(test.pop("21", None))
    #print test
    #mc.jsonConfigurations(test)


    #print dict(test)

    #mw.animateTrialID([0,1,3,2])
    #mw.animateTrialID([1])
    #mw.runTrial("lls.LLS",template,["t","theta","x","y","fx","fy"])
    #mw.runTrial("lls.LLS","lls.cfg",["t","x","fx","fy"])
