import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd
import random
from matplotlib import rc

#rc('font',**dict(family='serif',serif=['Computer Modern Roman'],size=16))
#font = {'family':'sans-serif',
        #'sans-serif':['Computer Modern Roman'],
#        'size':12}
#rc('font',**font)
#rc('text', usetex=True)

# control is blue, mass is red, inertia is green
style = {'c':'b','m':'r','i':'g'}

# make figures larger (12 in wide by 8 in tall)
plt.figure(1,figsize=(12,8))

ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())

saveDir = 'PerturbationPhaseLLSPersist'
modelName = 'Puck'
treatment = 'Phase'

offset = 283

varList = ['t','x','y','theta','fx','fy','dx','dy','dtheta','q','v','delta','omega','KE']

mw = model.ModelWrapper(saveDir)
mo = model.ModelOptimize(mw)
mc = model.ModelConfiguration(mw)
mp = modelplot.ModelPlot(mw)

mw.csvLoadData(mo.treatments.index)

controlObsIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
mw.csvLoadObs(controlObsIDs)
massObsIDs = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
mw.csvLoadObs(massObsIDs)
inertiaObsIDs = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124]
mw.csvLoadObs(inertiaObsIDs)

yLimits = {'x':(0,20), 'y':(-20,10), 'theta':(-1,1), 'v':(0,80), 'delta':(-3,3), 'omega':(-15,15),
'dx':(-80,80), 'dy':(-80,80), 'acc':(-2000,2000)}

template = mc.jsonLoadTemplate('templateControl')
dataStatsControl = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy', \
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975','acc','acc_025','acc_975'))
controlDataIDs = mo.treatments.query("Treatment == 'control'").index
dataControl = mp.combineData(controlDataIDs, offset)
#dataControl['KE'] = map(lambda x,y,z: .5*template['m']*(x**2 + y**2) + .5*template['I']*z**2, dataControl['Roach_vx'], dataControl['Roach_vy'], dataControl['Roach_dtheta'])
c = dataControl['DataID'].map(lambda x: x in controlDataIDs)
for sampleID in [int(sampleID) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]:
    d = dataControl['SampleID'].map(lambda x: x == sampleID)
    rowMean = dataControl[c][d].mean()
    rowQuantile025 = dataControl[c][d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = dataControl[c][d].quantile(q=[.975]).ix[.975]
    dataStatsControl.loc[sampleID] = \
    [rowMean['Roach_x'] \
    ,rowMean['Roach_y'] \
    ,rowMean['Roach_theta'] \
    ,rowMean['Roach_v'] \
    ,rowMean['Roach_heading'] \
    ,rowMean['Roach_dtheta'] \
    ,rowMean['Roach_vx'] \
    ,rowMean['Roach_vy'] \
    ,rowQuantile025['Roach_x'] \
    ,rowQuantile025['Roach_y'] \
    ,rowQuantile025['Roach_theta'] \
    ,rowQuantile025['Roach_v'] \
    ,rowQuantile025['Roach_heading'] \
    ,rowQuantile025['Roach_dtheta'] \
    ,rowQuantile025['Roach_vy'] \
    ,rowQuantile025['Roach_vx'] \
    ,rowQuantile975['Roach_x'] \
    ,rowQuantile975['Roach_y'] \
    ,rowQuantile975['Roach_theta'] \
    ,rowQuantile975['Roach_v'] \
    ,rowQuantile975['Roach_heading'] \
    ,rowQuantile975['Roach_dtheta'] \
    ,rowQuantile975['Roach_vx'] \
    ,rowQuantile975['Roach_vy'] \
    ,rowMean['CartAcceleration']
    ,rowQuantile025['CartAcceleration'] \
    ,rowQuantile975['CartAcceleration'] ]

obsStatsControl = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy', \
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975', 'cdf','acc','acc_025','acc_975'))
obsControl = mp.combineObs(controlObsIDs, 0)
#obsControl['delta'] = map(lambda x,y,z: np.arctan2(y, x)-z, obsControl['dx'], obsControl['dy'],obsControl['theta'])
obsControl['delta'] = map(lambda dx,dy,th: np.angle((dx + 1.j*dy) *np.exp(-1.j*th)), obsControl['dx'], obsControl['dy'],obsControl['theta'])
for sampleID in [int(sampleID) for sampleID in obsControl['SampleID'].unique()]:
    d = obsControl['SampleID'].map(lambda x: x == sampleID)
    rowMean = obsControl[d].mean()
    rowQuantile025 = obsControl[d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = obsControl[d].quantile(q=[.975]).ix[.975]
    obsStatsControl.loc[sampleID] = \
    [rowMean['x'] \
    ,rowMean['y'] \
    ,rowMean['theta'] \
    ,rowMean['v'] \
    ,rowMean['delta'] \
    ,rowMean['dtheta'] \
    ,rowMean['dx'] \
    ,rowMean['dy'] \
    ,rowQuantile025['x'] \
    ,rowQuantile025['y'] \
    ,rowQuantile025['theta'] \
    ,rowQuantile025['v'] \
    ,rowQuantile025['delta'] \
    ,rowQuantile025['dtheta'] \
    ,rowQuantile025['dx'] \
    ,rowQuantile025['dy'] \
    ,rowQuantile975['x'] \
    ,rowQuantile975['y'] \
    ,rowQuantile975['theta'] \
    ,rowQuantile975['v'] \
    ,rowQuantile975['delta'] \
    ,rowQuantile975['dtheta'] \
    ,rowQuantile975['dx'] \
    ,rowQuantile975['dy'] \
    ,float(obsControl[d]['q'].map(lambda x: x < 2).sum())/len(controlObsIDs) \
    ,-rowMean['accy']
    ,-rowQuantile025['accy'] \
    ,-rowQuantile975['accy'] ]

template = mc.jsonLoadTemplate('templateMass')
dataStatsMass = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy', \
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975', 'acc', 'acc_025','acc_975'))
massDataIDs = mo.treatments.query("Treatment == 'mass'").index
dataMass = mp.combineData(massDataIDs, offset)
c = dataMass['DataID'].map(lambda x: x in massDataIDs)
for sampleID in [int(sampleID) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]:
    d = dataMass['SampleID'].map(lambda x: x == sampleID)
    rowMean = dataMass[c][d].mean()
    rowQuantile025 = dataMass[c][d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = dataMass[c][d].quantile(q=[.975]).ix[.975]
    dataStatsMass.loc[sampleID] = \
    [rowMean['Roach_x'] \
    ,rowMean['Roach_y'] \
    ,rowMean['Roach_theta'] \
    ,rowMean['Roach_v'] \
    ,rowMean['Roach_heading'] \
    ,rowMean['Roach_dtheta'] \
    ,rowMean['Roach_vx'] \
    ,rowMean['Roach_vy'] \
    ,rowQuantile025['Roach_x'] \
    ,rowQuantile025['Roach_y'] \
    ,rowQuantile025['Roach_theta'] \
    ,rowQuantile025['Roach_v'] \
    ,rowQuantile025['Roach_heading'] \
    ,rowQuantile025['Roach_dtheta'] \
    ,rowQuantile025['Roach_vx'] \
    ,rowQuantile025['Roach_vy'] \
    ,rowQuantile975['Roach_x'] \
    ,rowQuantile975['Roach_y'] \
    ,rowQuantile975['Roach_theta'] \
    ,rowQuantile975['Roach_v'] \
    ,rowQuantile975['Roach_heading'] \
    ,rowQuantile975['Roach_dtheta'] \
    ,rowQuantile975['Roach_vx'] \
    ,rowQuantile975['Roach_vy'] \
    ,rowMean['CartAcceleration']
    ,rowQuantile025['CartAcceleration'] \
    ,rowQuantile975['CartAcceleration'] ]

obsStatsMass = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy', \
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975','cdf', 'acc', 'acc_025','acc_975'))
obsMass = mp.combineObs(massObsIDs, 0)
obsMass['delta'] = map(lambda x,y,z: np.arctan2(y, x)-z, obsMass['dx'], obsMass['dy'],obsMass['theta'])
for sampleID in [int(sampleID) for sampleID in obsMass['SampleID'].unique()]:
    d = obsMass['SampleID'].map(lambda x: x == sampleID)
    rowMean = obsMass[d].mean()
    rowQuantile025 = obsMass[d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = obsMass[d].quantile(q=[.975]).ix[.975]
    obsStatsMass.loc[sampleID] = \
    [rowMean['x'] \
    ,rowMean['y'] \
    ,rowMean['theta'] \
    ,rowMean['v'] \
    ,rowMean['delta'] \
    ,rowMean['dtheta'] \
    ,rowMean['dx'] \
    ,rowMean['dy'] \
    ,rowQuantile025['x'] \
    ,rowQuantile025['y'] \
    ,rowQuantile025['theta'] \
    ,rowQuantile025['v'] \
    ,rowQuantile025['delta'] \
    ,rowQuantile025['dtheta'] \
    ,rowQuantile025['dx'] \
    ,rowQuantile025['dy'] \
    ,rowQuantile975['x'] \
    ,rowQuantile975['y'] \
    ,rowQuantile975['theta'] \
    ,rowQuantile975['v'] \
    ,rowQuantile975['delta'] \
    ,rowQuantile975['dtheta'] \
    ,rowQuantile975['dx'] \
    ,rowQuantile975['dy'] \
    ,float(obsMass[d]['q'].map(lambda x: x < 2).sum())/len(massObsIDs) \
    ,-rowMean['accy']
    ,-rowQuantile025['accy'] \
    ,-rowQuantile975['accy'] ]

template = mc.jsonLoadTemplate('templateInertia')
dataStatsInertia = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy',\
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975', 'acc', 'acc_025','acc_975'))
inertiaDataIDs = mo.treatments.query("Treatment == 'inertia'").index
dataInertia = mp.combineData(inertiaDataIDs, offset)
c = dataInertia['DataID'].map(lambda x: x in inertiaDataIDs)
for sampleID in [int(sampleID) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]:
    d = dataInertia['SampleID'].map(lambda x: x == sampleID)
    rowMean = dataInertia[c][d].mean()
    rowQuantile025 = dataInertia[c][d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = dataInertia[c][d].quantile(q=[.975]).ix[.975]
    dataStatsInertia.loc[sampleID] = \
    [rowMean['Roach_x'] \
    ,rowMean['Roach_y'] \
    ,rowMean['Roach_theta'] \
    ,rowMean['Roach_v'] \
    ,rowMean['Roach_heading'] \
    ,rowMean['Roach_dtheta'] \
    ,rowMean['Roach_vx'] \
    ,rowMean['Roach_vy'] \
    ,rowQuantile025['Roach_x'] \
    ,rowQuantile025['Roach_y'] \
    ,rowQuantile025['Roach_theta'] \
    ,rowQuantile025['Roach_v'] \
    ,rowQuantile025['Roach_heading'] \
    ,rowQuantile025['Roach_dtheta'] \
    ,rowQuantile025['Roach_vx'] \
    ,rowQuantile025['Roach_vy'] \
    ,rowQuantile975['Roach_x'] \
    ,rowQuantile975['Roach_y'] \
    ,rowQuantile975['Roach_theta'] \
    ,rowQuantile975['Roach_v'] \
    ,rowQuantile975['Roach_heading'] \
    ,rowQuantile975['Roach_dtheta'] \
    ,rowQuantile975['Roach_vx'] \
    ,rowQuantile975['Roach_vy']
    ,rowMean['CartAcceleration']
    ,rowQuantile025['CartAcceleration'] \
    ,rowQuantile975['CartAcceleration'] ]

obsStatsInertia = pd.DataFrame(columns=('x','y','theta','v','delta','omega','dx','dy', \
    'x_025','y_025','theta_025','v_025','delta_025','omega_025','dx_025','dy_025', \
    'x_975','y_975','theta_975','v_975','delta_975','omega_975','dx_975','dy_975','cdf', 'acc', 'acc_025','acc_975'))
obsInertia = mp.combineObs(inertiaObsIDs, 0)
obsInertia['delta'] = map(lambda x,y,z: np.arctan2(y, x)-z, obsInertia['dx'], obsInertia['dy'],obsInertia['theta'])
for sampleID in [int(sampleID) for sampleID in obsInertia['SampleID'].unique()]:
    d = obsInertia['SampleID'].map(lambda x: x == sampleID)
    rowMean = obsInertia[d].mean()
    rowQuantile025 = obsInertia[d].quantile(q=[.025]).ix[.025]
    rowQuantile975 = obsInertia[d].quantile(q=[.975]).ix[.975]
    obsStatsInertia.loc[sampleID] = \
    [rowMean['x'] \
    ,rowMean['y'] \
    ,rowMean['theta'] \
    ,rowMean['v'] \
    ,rowMean['delta'] \
    ,rowMean['dtheta'] \
    ,rowMean['dx'] \
    ,rowMean['dy'] \
    ,rowQuantile025['x'] \
    ,rowQuantile025['y'] \
    ,rowQuantile025['theta'] \
    ,rowQuantile025['v'] \
    ,rowQuantile025['delta'] \
    ,rowQuantile025['dtheta'] \
    ,rowQuantile025['dx'] \
    ,rowQuantile025['dy'] \
    ,rowQuantile975['x'] \
    ,rowQuantile975['y'] \
    ,rowQuantile975['theta'] \
    ,rowQuantile975['v'] \
    ,rowQuantile975['delta'] \
    ,rowQuantile975['dtheta'] \
    ,rowQuantile975['dx'] \
    ,rowQuantile975['dy'] \
    ,float(obsInertia[d]['q'].map(lambda x: x < 2).sum())/len(inertiaObsIDs) \
    ,-rowMean['accy']
    ,-rowQuantile025['accy'] \
    ,-rowQuantile975['accy'] ]

timeData = [round(sampleID * .002,4) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsControl['SampleID'].unique() if sampleID >= 0]

plt.clf()
plt.figure(1,figsize=(12,8))
dataLegend, = plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x'],dataStatsControl[dataStatsControl.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
obsLegend, = plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x'],obsStatsControl[obsStatsControl.index <= 150]['y'],'b',linewidth=2.0,label=modelName + '_Mean')
plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x_025'],dataStatsControl[dataStatsControl.index <= 150]['y_025'],'k--',label='Roach_95%')
dataLegendS, = plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x_975'],dataStatsControl[dataStatsControl.index <= 150]['y_975'],'k--',label='Roach_95%')
obsLegendS, = plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x_025'],obsStatsControl[obsStatsControl.index <= 150]['y_025'],'b--',label=modelName + '_95%')
plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x_975'],obsStatsControl[obsStatsControl.index <= 150]['y_975'],'b--',label=modelName + '_95%')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('x (cm)')
plt.xlim(yLimits['x'][0],yLimits['x'][1])
plt.ylim(yLimits['y'][0],yLimits['y'][1])
plt.ylabel('y (cm)')
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': x vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-x_y')
plt.show()

plt.clf()
plt.figure(2,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['x'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['x_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['x_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['x'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['x_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['x_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['x'][0] + (yLimits['x'][1]-yLimits['x'][0]) * (1-obsStatsControl['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('x (cm)')
plt.ylim(yLimits['x'][0],yLimits['x'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs x', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_x')
plt.show()

plt.clf()
plt.figure(3,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['y'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['y_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['y_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['y'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['y_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['y_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['y'][0] + (yLimits['y'][1]-yLimits['y'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('y (cm/s)')
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_y')
plt.show()

plt.clf()
plt.figure(4,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['theta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['theta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['theta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['theta'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['theta_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['theta_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['theta'][0] + (yLimits['theta'][1]-yLimits['theta'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('theta (rad)')
plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_theta')
plt.show()

plt.clf()
plt.figure(5,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['v'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['v_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['v_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['v'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['v_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['v_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['v'][0] + (yLimits['v'][1]-yLimits['v'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('v (cm/s)')
plt.ylim(yLimits['v'][0],yLimits['v'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs v', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_v')
plt.show()

plt.clf()
plt.figure(6,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['delta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['delta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['delta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['delta'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['delta_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['delta_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['delta'][0] + (yLimits['delta'][1]-yLimits['delta'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('delta (rad)')
plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_delta')
plt.show()

plt.clf()
plt.figure(7,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['omega'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['omega_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['omega_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['omega'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['omega_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['omega_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['omega'][0] + (yLimits['omega'][1]-yLimits['omega'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('omega (rad/s)')
plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_omega')
plt.show()

plt.clf()
plt.figure(8,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['dx'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['dx_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['dx_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['dx'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['dx_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['dx_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dx'][0] + (yLimits['dx'][1]-yLimits['dx'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dx (cm/s)')
plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_dx')
plt.show()

plt.clf()
plt.figure(9,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['dy'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['dy_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsControl['dy_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['dy'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['dy_025'],'b--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsControl['dy_975'],'b--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dy'][0] + (yLimits['dy'][1]-yLimits['dy'][0]) * (1-obsStatsControl['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dy (cm/s)')
plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_dy')
plt.show()

plt.clf()
plt.figure(10,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsControl['acc'],'m',linewidth=2.0,label='Cart_Mean')
dataLegendS, = plt.plot(timeData,dataStatsControl['acc_025'],'m--',label='Cart_95%')
plt.plot(timeData,dataStatsControl['acc_975'],'m--',label='Cart_95%')
obsLegend, = plt.plot(timeObs,obsStatsControl['acc'],'y',linewidth=2.0,label='Model_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsControl['acc_025'],'y--',label='Model_95%')
plt.plot(timeObs,obsStatsControl['acc_975'],'y--',label='Model_95%')
plt.plot(timeObs, yLimits['acc'][0] + (yLimits['acc'][1]-yLimits['acc'][0]) * (1-obsStatsControl ['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('acc (cm2/s2)')
plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
#plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_o-t_acc')
plt.show()

timeData = [round(sampleID * .002,4) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsMass['SampleID'].unique() if sampleID >= 0]

plt.clf()
plt.figure(11,figsize=(12,8))
dataLegend, = plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x'],dataStatsMass[dataStatsMass.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
obsLegend, = plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x'],obsStatsMass[obsStatsMass.index <= 150]['y'],'r',linewidth=2.0,label=modelName + '_Mean')
dataLegendS, = plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x_025'],dataStatsMass[dataStatsMass.index <= 150]['y_025'],'k--',label='Roach_95%')
plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x_975'],dataStatsMass[dataStatsMass.index <= 150]['y_975'],'k--',label='Roach_95%')
obsLegendS, = plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x_025'],obsStatsMass[obsStatsMass.index <= 150]['y_025'],'r--',label=modelName + '_95%')
plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x_975'],obsStatsMass[obsStatsMass.index <= 150]['y_975'],'r--',label=modelName + '_95%')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.xlim(yLimits['x'][0],yLimits['x'][1])
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': x vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-x_y')
plt.show()

plt.clf()
plt.figure(12,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['x'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['x_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['x_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['x'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['x_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['x_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['x'][0] + (yLimits['x'][1]-yLimits['x'][0]) * (1-obsStatsMass['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('x (cm)')
plt.ylim(yLimits['x'][0],yLimits['x'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs x', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_x')
plt.show()

plt.clf()
plt.figure(13,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['y'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['y_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['y_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['y'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['y_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['y_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['y'][0] + (yLimits['y'][1]-yLimits['y'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('y (cm)')
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_y')
plt.show()

plt.clf()
plt.figure(14,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['theta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['theta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['theta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['theta'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['theta_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['theta_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['theta'][0] + (yLimits['theta'][1]-yLimits['theta'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('theta (rad)')
plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_theta')
plt.show()

plt.clf()
plt.figure(15,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['v'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['v_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['v_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['v'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['v_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['v_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['v'][0] + (yLimits['v'][1]-yLimits['v'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('v (cm/s)')
plt.ylim(yLimits['v'][0],yLimits['v'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs v', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_v')
plt.show()

plt.clf()
plt.figure(16,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['delta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['delta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['delta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['delta'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['delta_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['delta_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['delta'][0] + (yLimits['delta'][1]-yLimits['delta'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('delta (rad)')
plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_delta')
plt.show()

plt.clf()
plt.figure(17,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['omega'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['omega_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['omega_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['omega'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['omega_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['omega_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['omega'][0] + (yLimits['omega'][1]-yLimits['omega'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('omega (rad/s)')
plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_omega')
plt.show()

plt.clf()
plt.figure(18,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['dx'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['dx_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['dx_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['dx'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['dx_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['dx_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dx'][0] + (yLimits['dx'][1]-yLimits['dx'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dx (cm/s)')
plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_dx')
plt.show()

plt.clf()
plt.figure(19,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['dy'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['dy_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsMass['dy_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['dy'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['dy_025'],'r--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsMass['dy_975'],'r--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dy'][0] + (yLimits['dy'][1]-yLimits['dy'][0]) * (1-obsStatsMass['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dy (cm/s)')
plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_dy')
plt.show()

plt.clf()
plt.figure(20,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsMass['acc'],'m',linewidth=2.0,label='Cart_Mean')
dataLegendS, = plt.plot(timeData,dataStatsMass['acc_025'],'m--',label='Cart_95%')
plt.plot(timeData,dataStatsMass['acc_975'],'m--',label='Cart_95%')
obsLegend, = plt.plot(timeObs,obsStatsMass['acc'],'y',linewidth=2.0,label='Model_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsMass['acc_025'],'y--',label='Model_95%')
plt.plot(timeObs,obsStatsMass['acc_975'],'y--',label='Model_95%')
plt.plot(timeObs, yLimits['acc'][0] + (yLimits['acc'][1]-yLimits['acc'][0]) * (1-obsStatsMass ['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('acc (cm2/s2)')
plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
#plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-m_o-t_acc')
plt.show()

timeData = [round(sampleID * .002,4) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]
timeObs = [round(sampleID * .002,4) for sampleID in obsInertia['SampleID'].unique() if sampleID >= 0]

plt.clf()
plt.figure(21,figsize=(12,8))
dataLegend, = plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x'],dataStatsInertia[dataStatsInertia.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
obsLegend, = plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x'],obsStatsInertia[obsStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label=modelName + '_Mean')
dataLegendS, = plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x_025'],dataStatsInertia[dataStatsInertia.index <= 150]['y_025'],'k--',label='Roach_95%')
plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x_975'],dataStatsInertia[dataStatsInertia.index <= 150]['y_975'],'k--',label='Roach_95%')
obsLegendS, = plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x_025'],obsStatsInertia[obsStatsInertia.index <= 150]['y_025'],'g--',label=modelName + '_95%')
plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x_975'],obsStatsInertia[obsStatsInertia.index <= 150]['y_975'],'g--',label=modelName + '_95%')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.xlim(yLimits['x'][0],yLimits['x'][1])
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': x vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-x_y')
plt.show()

plt.clf()
plt.figure(22,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['x'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['x_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['x_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['x'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['x_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['x_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['x'][0] + (yLimits['x'][1]-yLimits['x'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('x (cm)')
plt.ylim(yLimits['x'][0],yLimits['x'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_x')
plt.show()

plt.clf()
plt.figure(23,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['y'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['y_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['y_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['y'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['y_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['y_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['y'][0] + (yLimits['y'][1]-yLimits['y'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('y (cm)')
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_y')
plt.show()

plt.clf()
plt.figure(24,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['theta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['theta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['theta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['theta'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['theta_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['theta_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['theta'][0] + (yLimits['theta'][1]-yLimits['theta'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('theta (rad)')
plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_theta')
plt.show()

plt.clf()
plt.figure(25,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['v'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['v_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['v_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['v'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['v_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['v_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['v'][0] + (yLimits['v'][1]-yLimits['v'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('v (cm/s)')
plt.ylim(yLimits['v'][0],yLimits['v'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_v')
plt.show()

plt.clf()
plt.figure(26,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['delta'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['delta_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['delta_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['delta'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['delta_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['delta_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['delta'][0] + (yLimits['delta'][1]-yLimits['delta'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('delta (rad)')
plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_delta')
plt.show()

plt.clf()
plt.figure(27,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['omega'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['omega_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['omega_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['omega'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['omega_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['omega_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['omega'][0] + (yLimits['omega'][1]-yLimits['omega'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('omega (rad/s)')
plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_omega')
plt.show()

plt.clf()
plt.figure(28,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['dx'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['dx_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['dx_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['dx'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['dx_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['dx_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dx'][0] + (yLimits['dx'][1]-yLimits['dx'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dx (cm/s)')
plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_dx')
plt.show()

plt.clf()
plt.figure(29,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['dy'],'k',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['dy_025'],'k--',label='Roach_95%')
plt.plot(timeData,dataStatsInertia['dy_975'],'k--',label='Roach_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['dy'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['dy_025'],'g--',label=modelName + '_95%')
plt.plot(timeObs,obsStatsInertia['dy_975'],'g--',label=modelName + '_95%')
plt.plot(timeObs, yLimits['dy'][0] + (yLimits['dy'][1]-yLimits['dy'][0]) * (1-obsStatsInertia['cdf']), color='0.5')
#plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dy (cm/s)')
plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_dy')
plt.show()

plt.clf()
plt.figure(30,figsize=(12,8))
dataLegend, = plt.plot(timeData,dataStatsInertia['acc'],'m',linewidth=2.0,label='Cart_Mean')
dataLegendS, = plt.plot(timeData,dataStatsInertia['acc_025'],'m--',label='Cart_95%')
plt.plot(timeData,dataStatsInertia['acc_975'],'m--',label='Cart_95%')
obsLegend, = plt.plot(timeObs,obsStatsInertia['acc'],'y',linewidth=2.0,label='Model_Mean')
obsLegendS, = plt.plot(timeObs,obsStatsInertia['acc_025'],'y--',label='Model_95%')
plt.plot(timeObs,obsStatsInertia['acc_975'],'y--',label='Model_95%')
plt.plot(timeObs, yLimits['acc'][0] + (yLimits['acc'][1]-yLimits['acc'][0]) * (1-obsStatsInertia ['cdf']), color='0.5')
plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('acc (cm2/s2)')
plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-i_o-t_acc')
plt.show()


################################################################################
################################################################################
################################################################################
################################################################################

timeObsControl = [round(sampleID * .002,4) for sampleID in obsControl['SampleID'].unique() if sampleID >= 0]

timeObsMass = [round(sampleID * .002,4) for sampleID in obsMass['SampleID'].unique() if sampleID >= 0]

timeObsInertia = [round(sampleID * .002,4) for sampleID in obsInertia['SampleID'].unique() if sampleID >= 0]


plt.clf()
plt.figure(31,figsize=(12,8))
obsLegend, = plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x'],obsStatsControl[obsStatsControl.index <= 150]['y'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x_025'],obsStatsControl[obsStatsControl.index <= 150]['y_025'],'b--',label=modelName + '_95%')
plt.plot(obsStatsControl[obsStatsControl.index <= 150]['x_975'],obsStatsControl[obsStatsControl.index <= 150]['y_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x'],obsStatsMass[obsStatsMass.index <= 150]['y'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x_025'],obsStatsMass[obsStatsMass.index <= 150]['y_025'],'r--',label=modelName + '_95%')
plt.plot(obsStatsMass[obsStatsMass.index <= 150]['x_975'],obsStatsMass[obsStatsMass.index <= 150]['y_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x'],obsStatsInertia[obsStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x_025'],obsStatsInertia[obsStatsInertia.index <= 150]['y_025'],'g--',label=modelName + '_95%')
plt.plot(obsStatsInertia[obsStatsInertia.index <= 150]['x_975'],obsStatsInertia[obsStatsInertia.index <= 150]['y_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.xlim(yLimits['x'][0],yLimits['x'][1])
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': x vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-x_y')
plt.show()

plt.clf()
plt.figure(32,figsize=(12,8))

obsLegendC, = plt.plot(timeObsControl,obsStatsControl['x'],'b',linewidth=2.0,label=modelName + '_Control_Mean')
obsLegendSC, = plt.plot(timeObsControl,obsStatsControl['x_025'],'b--',label=modelName + '_Control_95%')
plt.plot(timeObsControl,obsStatsControl['x_975'],'b--',label=modelName + '_Control_95%')

obsLegendM, = plt.plot(timeObsMass,obsStatsMass['x'],'r',linewidth=2.0,label=modelName + '_Mass_Mean')
obsLegendSM, = plt.plot(timeObsMass,obsStatsMass['x_025'],'r--',label=modelName + '_Mass_95%')
plt.plot(timeObsMass,obsStatsMass['x_975'],'r--',label=modelName + '_Mass_95%')

obsLegendI, = plt.plot(timeObsInertia,obsStatsInertia['x'],'g',linewidth=2.0,label=modelName + '_Inertia_Mean')
obsLegendSI, = plt.plot(timeObsInertia,obsStatsInertia['x_025'],'g--',label=modelName + '_Inertia_95%')
plt.plot(timeObsInertia,obsStatsInertia['x_975'],'g--',label=modelName + '_Inertia_95%')

plt.legend(handles=[obsLegendC, obsLegendSC, obsLegendM, obsLegendSM, obsLegendI, obsLegendSI], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('x (cm)')
plt.ylim(yLimits['x'][0],yLimits['x'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_x')
plt.show()

plt.clf()
plt.figure(33,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['y'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['y_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['y_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['y'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['y_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['y_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['y'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['y_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['y_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('y (cm)')
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_y')
plt.show()

plt.clf()
plt.figure(34,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['theta'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['theta_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['theta_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['theta'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['theta_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['theta_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['theta'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['theta_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['theta_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('theta (rad)')
plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_theta')
plt.show()

plt.clf()
plt.figure(35,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['v'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['v_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['v_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['v'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['v_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['v_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['v'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['v_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['v_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('v (cm/s)')
plt.ylim(yLimits['v'][0],yLimits['v'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_v')
plt.show()

plt.clf()
plt.figure(36,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['delta'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['delta_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['delta_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['delta'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['delta_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['delta_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['delta'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['delta_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['delta_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('delta (rad)')
plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_delta')
plt.show()

plt.clf()
plt.figure(37,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['omega'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['omega_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['omega_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['omega'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['omega_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['omega_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['omega'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['omega_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['omega_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('omega (rad/s)')
plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_omega')
plt.show()

plt.clf()
plt.figure(37,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['dx'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['dx_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['dx_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['dx'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['dx_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['dx_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['dx'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['dx_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['dx_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dx (cm/s)')
plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_dx')
plt.show()

plt.clf()
plt.figure(39,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['dy'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['dy_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['dy_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['dy'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['dy_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['dy_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['dy'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['dy_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['dy_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dy (cm/s)')
plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_dy')
plt.show()

plt.clf()
plt.figure(40,figsize=(12,8))

obsLegend, = plt.plot(timeObsControl,obsStatsControl['acc'],'b',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsControl,obsStatsControl['acc_025'],'b--',label=modelName + '_95%')
plt.plot(timeObsControl,obsStatsControl['acc_975'],'b--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsMass,obsStatsMass['acc'],'r',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsMass,obsStatsMass['acc_025'],'r--',label=modelName + '_95%')
plt.plot(timeObsMass,obsStatsMass['acc_975'],'r--',label=modelName + '_95%')

obsLegend, = plt.plot(timeObsInertia,obsStatsInertia['acc'],'g',linewidth=2.0,label=modelName + '_Mean')
obsLegendS, = plt.plot(timeObsInertia,obsStatsInertia['dy_025'],'g--',label=modelName + '_95%')
plt.plot(timeObsInertia,obsStatsInertia['acc_975'],'g--',label=modelName + '_95%')

#plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('acc (cm/s2)')
plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/'+modelName+'-c_m_i-t_acc')
plt.show()



################################################################################
################################################################################
################################################################################
################################################################################


timeDataControl = [round(sampleID * .002,4) for sampleID in dataControl['SampleID'].unique() if sampleID >= 0]

timeDataMass = [round(sampleID * .002,4) for sampleID in dataMass['SampleID'].unique() if sampleID >= 0]

timeDataInertia = [round(sampleID * .002,4) for sampleID in dataInertia['SampleID'].unique() if sampleID >= 0]


plt.clf()
plt.figure(41,figsize=(12,8))
dataLegend, = plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x'],dataStatsControl[dataStatsControl.index <= 150]['y'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x_025'],dataStatsControl[dataStatsControl.index <= 150]['y_025'],'b--',label='Roach_95%')
plt.plot(dataStatsControl[dataStatsControl.index <= 150]['x_975'],dataStatsControl[dataStatsControl.index <= 150]['y_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x'],dataStatsMass[dataStatsMass.index <= 150]['y'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x_025'],dataStatsMass[dataStatsMass.index <= 150]['y_025'],'r--',label='Roach_95%')
plt.plot(dataStatsMass[dataStatsMass.index <= 150]['x_975'],dataStatsMass[dataStatsMass.index <= 150]['y_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x'],dataStatsInertia[dataStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x_025'],dataStatsInertia[dataStatsInertia.index <= 150]['y_025'],'g--',label='Roach_95%')
plt.plot(dataStatsInertia[dataStatsInertia.index <= 150]['x_975'],dataStatsInertia[dataStatsInertia.index <= 150]['y_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.xlim(yLimits['x'][0],yLimits['x'][1])
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': x vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-x_y')
plt.show()

plt.clf()
plt.figure(42,figsize=(12,8))

dataLegendC, = plt.plot(timeDataControl,dataStatsControl['x'],'b',linewidth=2.0,label='Roach_Control_Mean')
dataLegendSC, = plt.plot(timeDataControl,dataStatsControl['x_025'],'b--',label='Roach_Control_95%')
plt.plot(timeDataControl,dataStatsControl['x_975'],'b--',label='Roach_95%')

dataLegendM, = plt.plot(timeDataMass,dataStatsMass['x'],'r',linewidth=2.0,label='Roach_Mass_Mean')
dataLegendSM, = plt.plot(timeDataMass,dataStatsMass['x_025'],'r--',label='Roach_Mass_95%')
plt.plot(timeDataMass,dataStatsMass['x_975'],'r--',label='Roach_95%')

dataLegendI, = plt.plot(timeDataInertia,dataStatsInertia['x'],'g',linewidth=2.0,label='Roach_Inertia_Mean')
dataLegendSI, = plt.plot(timeDataInertia,dataStatsInertia['x_025'],'g--',label='Roach_Inertia_95%')
plt.plot(timeDataInertia,dataStatsInertia['x_975'],'g--',label='Roach_95%')

plt.legend(handles=[dataLegendC, dataLegendSC, dataLegendM, dataLegendSM, dataLegendI, dataLegendSI], loc=2, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('x (cm)')
plt.ylim(yLimits['x'][0],yLimits['x'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_x')
plt.show()

plt.clf()
plt.figure(43,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['y'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['y_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['y_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['y'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['y_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['y_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['y'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['y_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['y_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('y (cm)')
plt.ylim(yLimits['y'][0],yLimits['y'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_y')
plt.show()

plt.clf()
plt.figure(44,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['theta'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['theta_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['theta_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['theta'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['theta_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['theta_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['theta'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['theta_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['theta_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('theta (rad)')
plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_theta')
plt.show()

plt.clf()
plt.figure(45,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['v'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['v_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['v_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['v'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['v_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['v_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['v'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['v_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['v_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('v (cm/s)')
plt.ylim(yLimits['v'][0],yLimits['v'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_v')
plt.show()

plt.clf()
plt.figure(46,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['delta'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['delta_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['delta_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['delta'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['delta_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['delta_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['delta'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['delta_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['delta_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('delta (rad)')
plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_delta')
plt.show()

plt.clf()
plt.figure(47,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['omega'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['omega_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['omega_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['omega'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['omega_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['omega_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['omega'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['omega_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['omega_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('omega (rad/s)')
plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_omega')
plt.show()

plt.clf()
plt.figure(48,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['dx'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['dx_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['dx_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['dx'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['dx_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['dx_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['dx'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['dx_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['dx_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dx (cm/s)')
plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_dx')
plt.show()

plt.clf()
plt.figure(49,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['dy'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['dy_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['dy_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['dy'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['dy_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['dy_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['dy'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['dy_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['dy_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('dy (cm/s)')
plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_dy')
plt.show()

plt.clf()
plt.figure(50,figsize=(12,8))

dataLegend, = plt.plot(timeDataControl,dataStatsControl['acc'],'b',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataControl,dataStatsControl['acc_025'],'b--',label='Roach_95%')
plt.plot(timeDataControl,dataStatsControl['acc_975'],'b--',label='Roach_95%')

dataLegend, = plt.plot(timeDataMass,dataStatsMass['acc'],'r',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataMass,dataStatsMass['acc_025'],'r--',label='Roach_95%')
plt.plot(timeDataMass,dataStatsMass['acc_975'],'r--',label='Roach_95%')

dataLegend, = plt.plot(timeDataInertia,dataStatsInertia['acc'],'g',linewidth=2.0,label='Roach_Mean')
dataLegendS, = plt.plot(timeDataInertia,dataStatsInertia['acc_025'],'g--',label='Roach_95%')
plt.plot(timeDataInertia,dataStatsInertia['acc_975'],'g--',label='Roach_95%')

#plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
plt.xlabel('t (s)')
plt.xlim(0,.3)
plt.ylabel('acc (cm/s2)')
plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
#plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
plt.grid(True)
ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())
plt.tight_layout()
plt.savefig('RobertFull/c_m_i_d-t_acc')
plt.show()
