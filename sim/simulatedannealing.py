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
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)
ms = modelplot.ModelStreamer()
stream1, stream2 = ms.createConnection()

varOpt = ['x','y','v','theta','omega']
INDEX_BEGIN = 0
INDEX_END = 198
MISSING_PENALTY = 100

testTrajectory = {'x': [], 'y': []}
xEnd = {'x':10.0,'y':-10.0}
diff = INDEX_END - INDEX_BEGIN
for i in range(INDEX_BEGIN,INDEX_END+1):
    testTrajectory['x'].append(float(i)*xEnd['x']/diff)
    testTrajectory['y'].append(float(i)*xEnd['y']/diff)

def checkIndex(observation, i):
    if any(observation.index == i):
        return True
    else:
        return False
'''
def timeVector():
    time = []
    currentTime = TIME_BEGIN
    while(currentTime <= TIME_END):
        time.append(currentTime)
        currentTime += TIME_STEP
        currentTime = round(currentTime,3)
    return time
'''

def cost(x0):
    summation = 0.0
    observation = mw.observations[mw.trialID()-1]
    for i in range(INDEX_BEGIN,INDEX_END+1):
        if checkIndex(observation,i):
            for var in xEnd:
                summation += float((testTrajectory[var][i] - observation.ix[i, var])**2)
        else:
            summation += MISSING_PENALTY
    return float(np.sqrt(summation))

def noChange(fun, x0, args, **options):
    return spo.OptimizeResult(x=x0, fun=fun(x0), success=True, nfev=1)

def streamTrial(ID):
    observation = mw.observations[ID]
    k=[]
    j=[]
    for i in range(INDEX_BEGIN,INDEX_END+1):
        if checkIndex(observation,i):
                k.append(observation.ix[i, 'x'])
                j.append(observation.ix[i, 'y'])

    stream1.write(dict(x=k, y=j))
    stream2.write(dict(x=testTrajectory['x'], y=testTrajectory['y']))

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
    mw.updateCost(ID, 10000.00)


if __name__ == "__main__":
    template = mc.jsonLoadTemplate('template')
    template = mw.packConfiguration(template)
    TIME_STEP = template['dt']
    ID = mw.runTrial("lls.LLS",mc.jsonSaveConfiguration(mc.generateConfiguration(template)),["t","theta","x","y","fx","fy","v","delta","omega"])
    #mw.animateTrialID([0, 1])

    minimizer_kwargs = {"method":noChange, "jac":False}
    xBegin = []
    for var in varOpt:
        xBegin.append(template[var])
    #ret = spo.basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,niter=200)
    try:
        ret = spo.basinhopping(cost,xBegin,minimizer_kwargs=minimizer_kwargs,niter=500,callback=modelSimulate,T=10.,stepsize=8.,interval=60)
        print ret
        print "Early: " + str(runtime[1])
        print "End: " + str(runtime[len(runtime)-1])
        best = pd.Series(mw.trials['Cost'])
        best.reset_index()
        bestOB = mw.observations[(best[best == min(best)].index)[0]]
        stream1.write(dict(x=list(bestOB['x']), y=list(bestOB['y'])))
    except KeyboardInterrupt:
        print "exit optimization"
