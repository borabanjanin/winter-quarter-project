import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy
import random



saveDir = 'StableOrbit'

#index0 and index1 are tuples

varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t', 'dx', 'dy', 'hx', 'hy']
#varList = ['t','x','y','theta','dx','dy','fx','fy','dtheta','delta','v','q']

mw = model.ModelWrapper(saveDir)

mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

mw.csvLoadData([0])

template = mc.jsonLoadTemplate("templateControl")

#template['dt'] = .002
#template['to'] = 0.0
#template['tf'] = 3.0
#template['N'] = 250
template1 = copy.copy(template)


mw.runTrial('lls.LLS', template, varList, 0, 'noAccel')


mw.saveTables()
