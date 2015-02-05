#!/usr/bin/python

import modelwrapper as model
import modelplot as modelplot

mw = model.ModelWrapper()

mc = model.ModelConfiguration(mw)
mc.jsonLoadTemplate('template')

variables = {}
mass = 2.9
I = 2.5

for i in range(-5,5):
    variables['x'] = 0
    variables['y'] = 0
    mc.jsonSaveConfiguration(mc.generateConfiguration(variables))

for confFile in mc.loadConfNames():
    mw.runTrial("lls.LLS",confFile,["t","theta","x","y","fx","fy"])

mp = modelplot.ModelPlot(mw)
mp.plotID(1,['x','theta','fx'])
