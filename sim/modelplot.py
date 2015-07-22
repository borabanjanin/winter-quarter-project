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
import pylab as plt
import matplotlib as mpl
import modelwrapper as model
from shrevz import tk
import os
import copy
pd.options.mode.chained_assignment = None

#import plotly.plotly as py
#from plotly.graph_objs import *# auto sign-in with credentials or use py.sign_in()
#import plotly.tools as tls
import time

'''
class ModelStreamer(object):
    def __init__(self):
        tls.set_credentials_file(stream_ids=[ \
        "ohm659z4y0", \
        "i3yxmjesit", \
        "qx4x099wpc", \
        "e0i69ts2oz"])

        self.stream_ids = tls.get_credentials_file()['stream_ids']
        self.streams = []

    def createConnection(self):
        stream1 = Stream(
            token=self.stream_ids[0],  # (!) link stream id to 'token' key
            maxpoints=80      # (!) keep a max of 80 pts on screen
        )

        stream2 = Stream(
            token=self.stream_ids[1],  # (!) link stream id to 'token' key
            maxpoints=80      # (!) keep a max of 80 pts on screen
        )

        trace1 = Scatter(
            x=[0],
            y=[0],
            mode='lines+markers',
            stream=stream1         # (!) embed stream id, 1 per trace
        )

        trace2 = Scatter(
            x=[0],
            y=[0],
            mode='lines+markers',
            stream=stream2         # (!) embed stream id, 1 per trace
        )

        data = Data([trace1, trace2])
        layout = Layout(title='Time Series')
        fig = Figure(data=data,layout=layout)
        unique_url = py.plot(fig, filename='test',auto_open=False)
        print unique_url

        stream1 = py.Stream(self.stream_ids[0])
        stream2 = py.Stream(self.stream_ids[1])
        stream1.open()
        stream2.open()
        self.streams.append(stream1)
        self.streams.append(stream2)
        return stream1,stream2

    def __del__(self):
        for stream in self.streams:
            stream.close
'''

class ModelPlot(object):
    def __init__(self, modelwrapper):
        self.figs = []
        self.pFigID = 0
        self.modelwrapper = modelwrapper

    def axisPlotVariable(self, ax, t, var,linestyle='r-.'):
        ax.plot(t,var,linestyle)
        ax.set_xlabel(t.name)
        ax.set_ylabel(var.name)
        return ax

    def saveFig(self,fig,savePath='test.png'):
        fig.savefig(savePath)

    def plotVar(self,figID,t,var,saveName):
        if figID == None:
            #TODO: implement protected id
            figID = self.pFigID
            self.pFigID += 1

        savePath = ''
        if saveName == None:
            savePath = os.path.join(self.modelwrapper.saveDir, var.name+'.png')
        else:
            savePath = os.path.join(self.modelwrapper.saveDir, saveName+'.png')

        fig = plt.figure(figID)
        ax = plt.subplot(111)
        ax = self.axisPlotVariable(ax,t,var)
        #fig.show()
        fig.savefig(savePath)
        #plt.close(fig)
        #self.figs.append(fig)

    def plotID(self,ID,vars,saveName=None):
        plotIDs = []
        for var in vars:
            self.plotVar(None, self.modelwrapper.observations[ID]['t'], \
            self.modelwrapper.observations[ID][var],saveName)

    def findVariableMax(self,variable):
        varList = []
        for ID in self.modelwrapper.observations:
            varList = varList + self.modelwrapper.observations[ID][variable].tolist()

        data = pd.Series(varList,index=range(len(varList)))
        return max(data)

    def findVariableMin(self,variable):
        varList = []
        for ID in self.modelwrapper.observations:
            varList = varList + self.modelwrapper.observations[ID][variable].tolist()

        data = pd.Series(varList,index=range(len(varList)))
        return min(data)

    def findObsOffset(self, observation, offsetMode):
        if isinstance(offsetMode, int):
            if len(observation.index) - 1 < offsetMode:
                return len(observation.index) - 1
            else:
                return offsetMode
        elif offsetMode == None:
            return 0
        else:
            raise Exception("ModelPlot: Improper offsetMode passed")

    '''    DEPRECIATED TO BE DELETED
    def combineObs(self, obsIDs, offsetMode):
        #self.modelwrapper.csvLoadObs(obsIDs)
        observations = self.modelwrapper.observations
        columnList = list(observations[obsIDs[0]].columns.values)
        columnList.insert(0,'SampleID')
        obsTable = pd.DataFrame(columns=columnList)
        for i in range(len(obsIDs)):
            obsID = obsIDs[i]
            offset = self.findObsOffset(observations[obsID], offsetMode)
            currentObsTable = pd.DataFrame(columns=columnList)
            for j in observations[obsID].index:
                sampleID = j-offset
                currentRow = list(observations[obsID].loc[j])
                currentRow.insert(0,sampleID)
                currentObsTable.loc[j] = currentRow
            obsTable = pd.concat([obsTable, currentObsTable])
        #self.modelwrapper.csvReleaseObs(obsIDs)
        return obsTable
    '''

    def combineObs(self, obsIDs, offsetMode):
        observations = self.modelwrapper.observations
        columnList = list(observations[obsIDs[0]].columns.values)
        columnList.insert(0,'SampleID')
        obsTable = pd.DataFrame(columns=columnList)
        for i in range(len(obsIDs)):
            obsID = obsIDs[i]
            observation = observations[obsID]
            newObservation = copy.deepcopy(observation)
            offset = self.findObsOffset(observation, offsetMode)
            newObservation['SampleID'] = map(lambda x : x - offset, observation.index)
            obsTable = pd.concat([obsTable, newObservation])
        return obsTable

    '''    DEPRECIATED TO BE DELETED
    def combineData2(self, dataIDs, offset):
        #self.modelwrapper.csvLoadData(dataIDs)
        data = self.modelwrapper.data
        columnList = list(data[dataIDs[0]].columns.values)
        columnList.insert(0,'SampleID')
        columnList.insert(1,'DataID')
        dataTable = pd.DataFrame(columns=columnList)
        for i in range(len(dataIDs)):
            dataID = dataIDs[i]
            beginIndex, endIndex = self.modelwrapper.findDataIndex(dataID, offset, 'all')
            currentDataTable = pd.DataFrame(columns=columnList)
            for j in range(beginIndex,endIndex+1):
                sampleID = int(j-offset)
                currentRow = list(data[dataID].loc[j])
                currentRow.insert(0,sampleID)
                currentRow.insert(1,dataID)
                currentDataTable.loc[j] = currentRow
            dataTable = pd.concat([dataTable, currentDataTable])
        #self.modelwrapper.csvReleaseData(dataIDs)
        return dataTable
    '''

    def combineData(self, dataIDs, offset):
        data = self.modelwrapper.data
        columnList = list(data[dataIDs[0]].columns.values)
        columnList.insert(0,'SampleID')
        columnList.insert(1,'DataID')
        dataTable = pd.DataFrame(columns=columnList)
        for i in range(len(dataIDs)):
            dataID = dataIDs[i]
            beginIndex, endIndex = self.modelwrapper.findDataIndex(dataID, offset, 'all')
            currentData = data[dataID].loc[beginIndex: endIndex]
            currentData['SampleID'] = map(lambda x : x - offset, currentData.index)
            currentData['DataID'] = dataID
            dataTable = pd.concat([dataTable, currentData])
        #self.modelwrapper.csvReleaseData(dataIDs)
        return dataTable

class ModelAnimate(object):
    """
    Used to plot simultaneous lls

    Usage:
        Pass data and plot upon init
    """

    """
    Global Constants
    """
    r = 1.01
    dd = 5*r

    '''
    Class Variables
    '''
    plotNum = 1

    def __init__(self,modelwrapper):
        self.observations = modelwrapper.observations
        self.modelwrapper = modelwrapper

    class Iterator(object):
        def __init__(self,IDs,plotID,animations,boundaries):
            self.IDs = IDs
            self.plotID = plotID
            self.animations = animations
            self.boundaries = boundaries

        def getIDs(self):
            return self.IDs

        def getPlotID(self):
            return self.plotID

        def getAnimations(self):
            return self.animations

        def getBoundaries(self):
            return self.boundaries

    @staticmethod
    def plotNumberInc(n=1):
        '''
        .plotNumberInc returns the current figure number and increments
        by default

        INPUTS:
          n - 1 x 1
            The size of the increment is by default 1
        '''
        value = ModelAnimate.plotNum
        ModelAnimate.plotNum += n
        return value

    def hasNext(self, iterator):
        '''
        .hasNext returns whether the iterator has another data point

        OUTPUTS:
            Boolean
        '''
        IDs = iterator.getIDs()
        animations = iterator.getAnimations()
        for ID in IDs:
            animation = animations[ID]
            if animation['index'] >= animation['length'] - 1:
                return False
        return True

    @staticmethod
    def Ellipse((x,y), (rx, ry), N=20, t=0, **kwargs):
        theta = 2*np.pi/(N-1)*np.arange(N)
        xs = x + rx*np.cos(theta)*np.cos(-t) - ry*np.sin(theta)*np.sin(-t)
        ys = y + rx*np.cos(theta)*np.sin(-t) + ry*np.sin(theta)*np.cos(-t)
        return xs, ys

    def checkIDs(self,IDs):
        for ID in IDs:
            if ID not in self.observations:
                raise Exception("ModelPlot: Not valid key given for observations: " + str(ID))

    def findOffset(self,IDs,boundaries):
        #TO DO:add xOffset logic
        xOffset = 0
        yOffset = 0

        for ID in IDs:
            values = boundaries[ID]
            values['yOffset'] = yOffset - values['my']
            values['xOffset'] = -values['mx']
            yOffset = values['yOffset'] + ModelAnimate.dd
            if(values['Mx'] > xOffset):
                xOffset = values['Mx']

        return (boundaries,(xOffset + ModelAnimate.dd,yOffset))


    def findBoundaries(self,IDs):
        boundaries = {}

        for ID in IDs:
            observation = self.observations[ID]
            values = {}
            values['mx'] = observation['x'].min()
            values['Mx'] = observation['x'].max()
            values['my'] = observation['y'].min()
            values['My'] = observation['y'].max()
            boundaries[ID] = values
            #print values['mx']
            #print values['my']
            #print values['Mx']
            #print values['My']

        return boundaries

    def generateFig(self,IDs):

        boundaries = self.findBoundaries(IDs)
        (boundaries,(xBoundary,yBoundary)) = self.findOffset(IDs,boundaries)
        plotID = ModelAnimate.plotNumberInc()
        sizeRatio = 10*xBoundary/yBoundary
        fig = plt.figure(plotID,figsize=(xBoundary/3,yBoundary/3))
        plt.clf()
        ax = fig.add_subplot(111,aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0-ModelAnimate.dd,xBoundary))
        ax.set_ylim((0-ModelAnimate.dd,yBoundary))
        return (boundaries,plotID,ax)

    def animGenerate(self,IDs):
        """
        .animGenerate  generates animation figure and data

        INPUTS:
            o - Obs - trajectory to animate
            dt - time step
        """
        self.checkIDs(IDs)
        (boundaries,plotID, ax) = self.generateFig(IDs)
        animations = {}
        #iterable['plotID'] = plotID

        for ID in IDs:
            observation = self.observations[ID]
            boundary = boundaries[ID]
            animation = {}
            animation['Lcom'], = ax.plot(observation['x'].loc[0]+boundary['xOffset'], \
            observation['y'].loc[0]+boundary['yOffset'], 'b.', ms=10.)
            animation['Ecom'], = ax.plot(*ModelAnimate.Ellipse((observation['x'].loc[0]+boundary['xOffset'],\
            observation['y'].loc[0]+boundary['yOffset']), \
            (ModelAnimate.r, 0.5*ModelAnimate.r), 20, observation['theta'].loc[0]))
            animation['Ecom'].set_linewidth(4.0)
            animation['Lft'],  = ax.plot([observation['x'].loc[0]+boundary['xOffset'],observation['fx'].loc[0]]+boundary['xOffset'], \
            [observation['y'].loc[0]+boundary['yOffset'],observation['fy'].loc[0]+boundary['yOffset']],'g.-',lw=4.)
            animation['index'] = 0
            animation['length'] = self.modelwrapper.dataLength(observation)
            animations[ID] = animation

        iterable = ModelAnimate.Iterator(IDs,plotID,animations,boundaries)
        return iterable


    def animIterate(self,iterator):
        boundaries = iterator.getBoundaries()
        animations = iterator.getAnimations()
        plotID = iterator.getPlotID()
        IDs = iterator.getIDs()

        for ID in IDs:
            animation = animations[ID]
            boundary = boundaries[ID]
            observation = self.observations[ID]
            i = animation['index']
            animation['Lcom'].set_xdata(observation['x'].loc[i]+boundary['xOffset'])
            animation['Lcom'].set_ydata(observation['y'].loc[i]+boundary['yOffset'])
            animation['Lft'].set_xdata([observation['x'].loc[i]+boundary['xOffset'],observation['fx'].loc[i]+boundary['xOffset']])
            animation['Lft'].set_ydata([observation['y'].loc[i]+boundary['yOffset'],observation['fy'].loc[i]+boundary['yOffset']])
            Ex,Ey = ModelAnimate.Ellipse((observation['x'].loc[i]+boundary['xOffset'],observation['y'].loc[i]+boundary['yOffset']), \
            (0.5*ModelAnimate.r, ModelAnimate.r), t=observation['theta'].loc[i])
            animation['Ecom'].set_xdata(Ex)
            animation['Ecom'].set_ydata(Ey)
            animation['index'] = i + 1

        plt.figure(plotID).canvas.draw()


    def animateTrialID(self,IDs):
        #ma = mp.ModelAnimate(self.observations)
        self.modelwrapper.csvLoadObs(IDs)
        iterator = self.animGenerate(IDs)
        while self.hasNext(iterator):
            time.sleep(0.01)
            self.animIterate(iterator)
        self.modelwrapper.csvReleaseObs(IDs)

if __name__ == "__main__":
    mw = model.ModelWrapper('BestTrials-1k')
    mp = ModelPlot(mw)
    #mp.combineObs([0,1],283)
    mp.combineData([0,1],283)
    #ma = ModelAnimate(mw.observations)
    #ma.animateTrialID([0, 1])
    #print "happy"
