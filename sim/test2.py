import modelwrapper as mp
import matplotlib.pyplot as plt
import os
import numpy as np
from util import num

mw = mp.ModelWrapper()
mo = mp.ModelOptimize(mw)
mc = mp.ModelConfiguration(mw)


mw.csvLoadData([0])


mw.runTrial("lls.LLStoPuck",mc.jsonLoadTemplate("config-360"),mp.ModelOptimize.parmList, 0)

beginIndex, endIndex = mw.findDataIndex(0, None, 'all')

#accel = ( lambda s,i,j : np.array([0.,num.interp1(s,t,self.data[dataID]),0.]) )



#mw.saveTables()
'''
print num.interp1(np.array([0.2]),timeData,accelData)

#print beginIndex
#print endIndex



assert np.all(np.diff(timeData) >= 0)
'''
