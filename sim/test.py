import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time

saveDir = 'BestTrials-1k'

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc2 = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

dataTable =  mp.combineData([0, 1, 2], 283)
print dataTable.query('SampleID == 0')['CartAcceleration']
#obsTable = mp.combineObs([0,1,2], 283)

# for Sam's sake
#D = np.dstack([d.query('SampleID == %d'%j) for j in range(-90,0)]).swapaxes(0,2)

# TODO: this should look qualitatively like Fig 4.B in RevzenBurden2013.pdf
#d = dataTable
#plt.plot(np.asarray([d.query('SampleID == %d'%j)['CartAcceleration'].mean() for j in range(-100,200)]))
