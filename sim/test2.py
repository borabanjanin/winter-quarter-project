import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np


saveDir = 'StableOrbit'

varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

mw.csvLoadData([0])

templateControl = mc.jsonLoadTemplate("templateControl")
templateMass = mc.jsonLoadTemplate("templateMass")
templateInertia = mc.jsonLoadTemplate("templateInertia")

templateControl['tf'] = 5.0
templateMass['tf'] = 5.0
templateInertia['tf'] = 5.0
templateControl['N'] = 250
templateMass['N'] = 250
templateInertia['N'] = 250

mw.runTrial('lls.LLS', templateControl, varList, 0)
mw.runTrial('lls.LLS', templateMass, varList, 0)
mw.runTrial('lls.LLS', templateInertia, varList, 0)

mw.saveTables()
