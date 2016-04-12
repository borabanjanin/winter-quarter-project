import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy
import random
import lls



saveDir = 'StableOrbitHighRes'

#index0 and index1 are tuples

varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t', 'dx', 'dy', 'hx', 'hy']
#varList = ['t','x','y','theta','dx','dy','fx','fy','dtheta','delta','v','q']

mw = model.ModelWrapper(saveDir)

mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

dataIDs = mo.treatments.query("Treatment == 'control'").index
mw.csvLoadData(dataIDs)

template = mc.jsonLoadTemplate("templateInertia")

template['dt'] = .0001
template['to'] = 0.0
template['tf'] = 1.0
template['N'] = 250

mw.runTrial('lls.LLS', template, varList, 0, 'noAccel')
mw.saveTables()

'''
for i in dataIDs:
    print i
    mw.runTrial('lls.LLS', template, varList, dataID=i, accel='dataAccel')

mw.saveTables()
'''
