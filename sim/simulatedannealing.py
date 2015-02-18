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

import modelwrapper as model
import modelplot as modelplot

import pandas as pd
import numpy as np
import scipy.optimize as spo

import time
START_TIME = time.time()
runtime = []

mw = model.ModelWrapper()
DATA_ID = 1
mw.csvLoadData(DATA_ID)
optimizationData = mw.data[DATA_ID]

mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

ms = modelplot.ModelStreamer()
stream1, tream2 = ms.createConnection()
#Variables to adjust
varOpt = ['x','y','v','theta','omega']

#Variables for cost function
varCost=['x','y']

label = {'x':'Roach_x','y':'Roach_y','v':'Roach_v','theta':'Roach_theta','omega':'Roach_dtheta'}

DATA_OFFSET = 283

INDEX_BEGIN = 0
INDEX_END = 200

#Penalty for missing a data point per index
MISSING_PENALTY = 100

'''
#Create test trajectory
testTrajectory = {'x': [], 'y': []}
xEnd = {'x':12.0,'y':-12.0}
diff = INDEX_END - INDEX_BEGIN
for i in range(INDEX_BEGIN,INDEX_END+1):
    testTrajectory['x'].append(float(i)*xEnd['x']/diff)
    testTrajectory['y'].append(float(i)*xEnd['y']/diff)
'''

#Check if index exists
def checkModelIndex(observation, i):
    if any(observation.index == i):
        return True
    else:
        return False

def checkDataIndex(data, i, variables):
    for variables in variables:
        if isnull(data.ix[i,variable]):
            raise Exception("SimulatedAnealing: Null data " + str(i) + variable)
        else:
            return True

#Calculate cost for actual versus
#expected trajectory values
#Least squares error
def cost(x0):
    summation = 0.0
    observation = mw.observations[mw.trialID()-1]
    for i in range(INDEX_BEGIN,INDEX_END+1):
        if checkModelIndex(observation,i):
            currentDiff = []
            for var in varCost:
                dataName = label[var]
                summation += (float(optimizationData.ix[i + DATA_OFFSET, dataName]) - float(observation.ix[i, var]))**2
        else:
            summation += MISSING_PENALTY
    return summation

#The minimizer just returns values
def noChange(fun, x0, args, **options):
    return spo.OptimizeResult(x=x0, fun=fun(x0), success=True, nfev=1)

#Used to stream data to plotly
def streamTrial(ID):
    observation = mw.observations[ID]
    k=[]
    j=[]
    h=[]
    l=[]
    for i in range(INDEX_BEGIN,INDEX_END+1):
        if checkModelIndex(observation,i):
                k.append(observation.ix[i, 'x'])
                j.append(observation.ix[i, 'y'])
                h.append(optimizationData.ix[i + DATA_OFFSET, label['x']])
                l.append(optimizationData.ix[i + DATA_OFFSET, label['y']])

    stream1.write(dict(x=k, y=j))
    stream2.write(dict(x=h, y=l))

#Used in basinhopping callback
#ModelWrapper hook
def modelSimulate(x, f, record):
    global START_TIME
    global runtime
    print f
    runtime.append(time.time() - START_TIME)
    START_TIME = time.time()
    variables = {}
    for i in range(x.size):
        variables[varOpt[i]] = x[i]
    confID = mc.generateConfiguration(variables)
    confFile = mc.jsonSaveConfiguration(confID)
    ID = mw.runTrial("lls.LLS",confFile,["t","theta","x","y","fx","fy","v","delta"])
    if ID % 20 == 0:
        streamTrial(ID)
    mw.updateCost(ID-1, f)
    mw.updateDataID(ID,DATA_ID)
    mw.updateCost(ID, 1000000.00)


if __name__ == "__main__":
    template = mc.jsonLoadTemplate('template')
    template = mw.packConfiguration(template)
    #TIME_STEP = template['dt']
    ID = mw.runTrial("lls.LLS",mc.jsonSaveConfiguration(mc.generateConfiguration(template)),["t","theta","x","y","fx","fy","v","delta","omega"])
    #mw.animateTrialID([0, 1])

    minimizer_kwargs = {"method":noChange, "jac":False}
    xBegin = []
    for var in varOpt:
        xBegin.append(template[var])
    try:
        ret = spo.basinhopping(cost,xBegin,minimizer_kwargs=minimizer_kwargs,niter=100,callback=modelSimulate,T=10.,stepsize=8.,interval=60)
        print ret

        #Shows run time
        print "Early: " + str(runtime[1])
        print "End: " + str(runtime[len(runtime)-1])

        #Displays best trial at the end
        best = pd.Series(mw.trials['Cost'])
        best.reset_index()
        bestOB = mw.observations[(best[best == min(best)].index)[0]]
        stream1.write(dict(x=list(bestOB['x']), y=list(bestOB['y'])))

    except KeyboardInterrupt:
        print "exit optimization"
