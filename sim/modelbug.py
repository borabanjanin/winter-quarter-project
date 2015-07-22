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
mw.csvLoadObs([0])

plt.figure(1); plt.clf();

plotVars = ['y','x','q']
N = len(plotVars)
n = 1

def plotDetrend(var):
  global n

  if var not in ['fx','fy','t']:
    tO = mw.observations[0][var].dropna()
    tc = np.polyfit(x=tO.index, y=tO, deg=1)
    signal = tO - (tO.index * tc[0] + tc[1])
  else:
    signal = mw.observations[0][var].dropna()

  ax = plt.subplot(N,1,n); ax.grid('on')
  ax.plot(signal, '.-')
  ax.set_ylabel(var)
  n += 1

for var in plotVars:
  plotDetrend(var)

plt.show()
