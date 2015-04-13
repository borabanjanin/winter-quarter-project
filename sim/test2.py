import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np


#saveDir = 'BestTrials-1k'

varList = ['x','y','theta','fx','fy','dx','dy','dtheta','q']

mw = model.ModelWrapper()
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)

mw.csvLoadData([0])

oddPosition =  mw.data[0]["TarsusBody1_x"] + mw.data[0]["TarsusBody3_x"] + mw.data[0]["TarsusBody5_x"]
evenPosition =  mw.data[0]["TarsusBody2_x"] + mw.data[0]["TarsusBody4_x"] + mw.data[0]["TarsusBody6_x"]
oddV =  mw.data[0]["TarsusBody1_vx"] + mw.data[0]["TarsusBody3_vx"] + mw.data[0]["TarsusBody5_vx"]
evenV =  mw.data[0]["TarsusBody2_vx"] + mw.data[0]["TarsusBody4_vx"] + mw.data[0]["TarsusBody6_vx"]

pos = oddPosition - evenPosition
vel = oddV - evenV

xv = pos + 1.j * vel

plt.clf()
plt.figure(1)
#plt.plot(pos[:283], vel[:283], lw=4)
plt.plot(mw.data[0]['Roach_xv_phase'])
