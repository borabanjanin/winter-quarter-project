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
from opt import opt

import os
import time
import sys
import json

class ModelWrapper(object):
    """
    Used to run and store simulation results

    Usage:

    """

    """
    Global Constants
    """
    NUMBER_MODELS = 1000

    def __init__(self):
        self.observations = {}
        self.parameters = {}
        self.configurations = {}
        #self.trials = pd.DataFrame(columns=('ModelName','HashParm','HashConf'),index=range(self.NUMBER_MODELS))
        self.trials = pd.DataFrame(columns=('ModelName','HashParm','HashConf'))
        self.loadTables()
        #TO DO: FIND THE CORRECT STARTING KEY VALUE
        self.ptrialID = 0

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

    def trialID(self):
        '''
        .trialID returns the current trialID.

        OUTPUTS:
            int
        '''
        return self.ptrialID

    def incTrialID(self):
        '''
        .incTrialID increments the trialID.
        '''
        self.ptrialID += 1

        if self.ptrialID == sys.maxint:
            raise Exception("ModelWrapper: No new keys available")

    def newTrialID(self):
        '''
        .newTrialID returns a new trialID

        OUTPUTS:
            int
        '''
        self.incTrialID()
        return self.trialID()-1

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

        if(length == sys.maxint):
            length = 0

        return length

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

        self.parameters[ID] = listParm

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

        self.configurations[ID] = p.items()
        #Check if you can make dictionary dictionary

    def storeMod(self, ID, modelName):
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
        confHash = self.hashaList(self.configurations[ID])
        #TO DO: Reliablity of hashing on numbers, neccessary?
        self.trials.loc[ID] = [modelName, parmHash,confHash]

    def runModel(self, modelName=None, config=None, listParm=None):
        '''
        .runModel takes the model call, configuration file, and list of
        parameters to be tracked. It then runs the model and stores the
        observations, parameters tracked, and configurations for the respective
        trial.

        INPUTS:
            ID - 1 x 1 int
            p - 1 x n list

        OUTPUTS:
            parameters - m x n dictionary
        '''
        op = opt.Opt()

        if modelName == None \
        or config == None \
        or listParm == None:
            raise Exception("ModelWrapper: No model name, config file, or list of variables to track was given")

        configPath = os.getcwd() + "/" + config

        if os.path.isfile(configPath) == False:
            raise Exception("ModelWrapper: Not a valid config file")

        op.pars(fi=config)
        p = op.p

        model = eval(modelName)(p['dt'])
        z = model.ofind(np.asarray(p['z0']),(p['q0'],),p['N'],modes=[1])
        #t1,x1,q1 = lls1(0, 1e99, x01, q01, N1, dbg=dbg1)
        #TO DO: add model initialization method, configuration
        self.printGaitStats(z)
        #TO DO: add model initialization method, configuration
        o = model.obs().resample(p['dt'])
        ID = self.newTrialID()
        self.storeObs(ID, o, listParm)
        self.storeParms(ID,listParm)
        self.storeConf(ID,p)
        self.storeMod(ID, modelName)

    def csvTrials(self):
        self.trials.to_csv('trials.csv', sep='\t')

    def csvObservations(self):
        for key in self.observations.keys():
            observationsDF = self.createDataFrame(self.checkParmLength(self.observations[key]), self.observations[key])
            csvPath = 'observations/observation' + str(key) + '.csv'
            observationsDF.to_csv(csvPath, sep='\t')

    def csvConfigurations(self):
        with open('configurations.json', 'w') as file:
                json.dump(self.configurations, file)
                
    def csvParameters(self):
        with open('parameters.json', 'w') as file:
                json.dump(self.parameters, file)

    def saveTables(self):
        self.csvTrials()
        self.csvObservations()
        self.csvConfigurations()
        self.csvParameters()

    def __del__(self):
        self.saveTables()

    def csvLoadTrials(self):
        self.trials.from_csv('trials.csv', sep='\t')
        print self.trials

    #TO DO:
    def loadTables(self):
        self.csvLoadTrials()
    #function to load each table
    #function to load keys

if __name__ == "__main__":
    mw = ModelWrapper()
    mw.runModel("lls.LLS","lls.cfg",["t","x","fx"])
    #mw.runModel("lls.LLS","lls.cfg",["t","x","fx","fy"])
    #print mw.configurations[1]
