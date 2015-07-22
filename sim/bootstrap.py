import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd
import random
import copy
from matplotlib import rc

# control is blue, mass is red, inertia is green
style = {'c':'b','m':'r','i':'g','d':'k'}

# set the y limits for the plots
yLimits = {'x':(0,20), 'y':(-20,10), 'theta':(-1,1), 'v':(0,80), 'delta':(-3,3), 'omega':(-15,15),
'dx':(-80,80), 'dy':(-80,80), 'acc':(-2000,2000)}

# make figures larger (12 in wide by 8 in tall)
plt.figure(1,figsize=(12,8))

ax = plt.gca()
ax.set_xticklabels(ax.get_xticks())
ax.set_yticklabels(ax.get_yticks())

saveDir = 'PerturbationPhaseLLSPersist'
modelName = 'LLSPersist'
treatment = 'Phase'

offset = 283
samples = 200

varList = ['x','y','theta','v','delta','omega','dx','dy', 'acc']

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
'dx':(-80,80), 'dy':(-80,80), 'acc':(-2000,2000), 'accX':(-0.25,0.25), 'accY':(-0.25,0.25)}

def dataTableStats(data, dataIDs, columnList):
    dataStats = pd.DataFrame(columns=columnList)
    for sampleID in [int(sampleID) for sampleID in data['SampleID'].unique() if sampleID >= -25]:
        c = data['SampleID'].map(lambda x: x == sampleID)
        rowMean = data[c].mean()
        dataStats.loc[sampleID] = \
        [sampleID \
        ,0.0 \
        ,rowMean['Roach_x'] \
        ,rowMean['Roach_y'] \
        ,rowMean['Roach_theta'] \
        ,rowMean['Roach_v'] \
        ,rowMean['Roach_heading'] \
        ,rowMean['Roach_dtheta'] \
        ,rowMean['Roach_vx'] \
        ,rowMean['Roach_vy'] \
        ,rowMean['CartAcceleration']]
    return dataStats

def dataBootStrap(dataIDs, samples ,columnList):
    random.seed(2)
    columnList = copy.deepcopy(columnList)
    columnList.insert(0,'SampleID')
    columnList.insert(1,'MeanID')
    bootstrap = pd.DataFrame(columns=columnList)
    for i in range(samples):
        randomDataIDs = np.random.choice(dataIDs, len(dataIDs))
        data = mp.combineData(randomDataIDs, offset)
        dataStats = dataTableStats(data, randomDataIDs, columnList)
        dataStats['MeanID'] = i
        bootstrap = pd.concat([bootstrap, dataStats])
    return bootstrap

def obsTableStats(observations, obsIDs, columnList):
    obsStats = pd.DataFrame(columns=columnList)
    observations['delta'] = map(lambda dx,dy,th: np.angle((dx + 1.j*dy) *np.exp(-1.j*th)), observations['dx'], observations['dy'],observations['theta'])
    for sampleID in [int(sampleID) for sampleID in observations['SampleID'].unique()]:
        d = observations['SampleID'].map(lambda x: x == sampleID)
        rowMean = observations[d].mean()
        #rowQuantile025 = obsControl[d].quantile(q=[.025]).ix[.025]
        #rowQuantile975 = obsControl[d].quantile(q=[.975]).ix[.975]
        obsStats.loc[sampleID] = \
        [sampleID \
        ,0 \
        ,rowMean['x'] \
        ,rowMean['y'] \
        ,rowMean['theta'] \
        ,rowMean['v'] \
        ,rowMean['delta'] \
        ,rowMean['dtheta'] \
        ,rowMean['dx'] \
        ,rowMean['dy']
        ,rowMean['accy']]
    return obsStats

def obsBootStrap(obsIDs, samples ,columnList):
    random.seed(2)
    columnList = copy.deepcopy(columnList)
    bootstrap = pd.DataFrame(columns=columnList)
    columnList.insert(0,'SampleID')
    columnList.insert(1,'MeanID')
    for i in range(samples):
        randomObsIDs = np.random.choice(obsIDs, len(obsIDs))
        observations = mp.combineObs(randomObsIDs, 0)
        obsStats = obsTableStats(observations, randomObsIDs, columnList)
        obsStats['MeanID'] = i
        bootstrap = pd.concat([bootstrap, obsStats])
    return bootstrap

def dataTableStatsBootstrap(data):
    dataStats = pd.DataFrame(columns=('SampleID','x','y','theta','v','delta','omega','dx','dy', \
        'x_01','y_01','theta_01','v_01','delta_01','omega_01','dx_01','dy_01', \
        'x_99','y_99','theta_99','v_99','delta_99','omega_99','dx_99','dy_99','acc','acc_01','acc_99'))

    for sampleID in [int(sampleID) for sampleID in data['SampleID'].unique() if sampleID >= -25]:
        c = data['SampleID'].map(lambda x: x == sampleID)
        rowMean = data[c].mean()
        rowQuantile01 = data[c].quantile(q=[.01]).ix[.01]
        rowQuantile99 = data[c].quantile(q=[.99]).ix[.99]
        dataStats.loc[sampleID] = \
        [sampleID \
        ,rowMean['x'] \
        ,rowMean['y'] \
        ,rowMean['theta'] \
        ,rowMean['v'] \
        ,rowMean['delta'] \
        ,rowMean['omega'] \
        ,rowMean['dx'] \
        ,rowMean['dy'] \
        ,rowQuantile01['x'] \
        ,rowQuantile01['y'] \
        ,rowQuantile01['theta'] \
        ,rowQuantile01['v'] \
        ,rowQuantile01['delta'] \
        ,rowQuantile01['omega'] \
        ,rowQuantile01['dx'] \
        ,rowQuantile01['dy'] \
        ,rowQuantile99['x'] \
        ,rowQuantile99['y'] \
        ,rowQuantile99['theta'] \
        ,rowQuantile99['v'] \
        ,rowQuantile99['delta'] \
        ,rowQuantile99['omega'] \
        ,rowQuantile99['dx'] \
        ,rowQuantile99['dy'] \
        ,rowMean['acc']
        ,rowQuantile01['acc'] \
        ,rowQuantile99['acc'] ]

    return dataStats

def obsTableStatsBootstrap(observations):
    obsStats = pd.DataFrame(columns=('SampleID','x','y','theta','v','delta','omega','dx','dy', \
        'x_01','y_01','theta_01','v_01','delta_01','omega_01','dx_01','dy_01', \
        'x_99','y_99','theta_99','v_99','delta_99','omega_99','dx_99','dy_99','acc','acc_01','acc_99'))

    for sampleID in [int(sampleID) for sampleID in observations['SampleID'].unique() if sampleID >= -25]:
        c = observations['SampleID'].map(lambda x: x == sampleID)
        rowMean = observations[c].mean()
        rowQuantile01 = observations[c].quantile(q=[.01]).ix[.01]
        rowQuantile99 = observations[c].quantile(q=[.99]).ix[.99]
        obsStats.loc[sampleID] = \
        [sampleID \
        ,rowMean['x'] \
        ,rowMean['y'] \
        ,rowMean['theta'] \
        ,rowMean['v'] \
        ,rowMean['delta'] \
        ,rowMean['omega'] \
        ,rowMean['dx'] \
        ,rowMean['dy'] \
        ,rowQuantile01['x'] \
        ,rowQuantile01['y'] \
        ,rowQuantile01['theta'] \
        ,rowQuantile01['v'] \
        ,rowQuantile01['delta'] \
        ,rowQuantile01['omega'] \
        ,rowQuantile01['dx'] \
        ,rowQuantile01['dy'] \
        ,rowQuantile99['x'] \
        ,rowQuantile99['y'] \
        ,rowQuantile99['theta'] \
        ,rowQuantile99['v'] \
        ,rowQuantile99['delta'] \
        ,rowQuantile99['omega'] \
        ,rowQuantile99['dx'] \
        ,rowQuantile99['dy'] \
        ,-rowMean['acc']
        ,-rowQuantile01['acc'] \
        ,-rowQuantile99['acc'] ]

    return obsStats


#CREATE DATA TABLES
#dataIDs = mo.treatments.query("Treatment == 'control'").index
#bootstrapData = dataBootStrap(dataIDs, samples, varList)
#bootstrapData.to_csv('Bootstrap/ControlBootstrap.csv', sep='\t')
#dataIDs = mo.treatments.query("Treatment == 'mass'").index
#bootstrapData = dataBootStrap(dataIDs, samples, varList)
#bootstrapData.to_csv('Bootstrap/MassBootstrap.csv', sep='\t')
#dataIDs = mo.treatments.query("Treatment == 'inertia'").index
#bootstrapData = dataBootStrap(dataIDs, samples, varList)
#bootstrapData.to_csv('Bootstrap/InertiaBootstrap.csv', sep='\t')

#CREATE OBS TABLES
#mw.csvLoadObs(controlObsIDs)
#bootstrapObs = obsBootStrap(controlObsIDs, samples, varList)
#bootstrapObs.to_csv('Bootstrap/Control' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t')
#mw.csvLoadObs(massObsIDs)
#bootstrapObs = obsBootStrap(massObsIDs, samples, varList)
#bootstrapObs.to_csv('Bootstrap/Mass' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t')
#mw.csvLoadObs(inertiaObsIDs)
#bootstrapObs = obsBootStrap(inertiaObsIDs, samples, varList)
#bootstrapObs.to_csv('Bootstrap/Inertia' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t')


#READ DATA TABLES
#bootstrapDataStatsControl = dataTableStatsBootstrap(pd.read_csv('Bootstrap/ControlBootstrap.csv', sep='\t'))
#bootstrapDataStatsMass = dataTableStatsBootstrap(pd.read_csv('Bootstrap/MassBootstrap.csv', sep='\t'))
#bootstrapDataStatsInertia = dataTableStatsBootstrap(pd.read_csv('Bootstrap/InertiaBootstrap.csv', sep='\t'))

#READ OBS TABLES
#bootstrapObsStatsControl = obsTableStatsBootstrap(pd.read_csv('Bootstrap/Control' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t', index_col=0))
#bootstrapObsStatsMass = obsTableStatsBootstrap(pd.read_csv('Bootstrap/Mass' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t', index_col=0))
#bootstrapObsStatsInertia = obsTableStatsBootstrap(pd.read_csv('Bootstrap/Inertia' + str(modelName) + str(treatment) + 'Bootstrap.csv', sep='\t', index_col=0))

'''
def plotData(m1, l1, u1, m2, l2, u2, color1, color2, figNum, xLabel, yLabel, limits, saveName):
    plt.clf()
    plt.figure(figNum,figsize=(12,8))

    dataLegend, = plt.plot(m1[0],m1[1],color1,linewidth=2.0,label=modelName + '_Mean')
    dataLegendS, = plt.plot(l1[0],l1[1],color1+'--',label=modelName + '_95%')
    plt.plot(u1[0],u1[1],color1+'--',label=modelName + '_95%')

    obsLegend, = plt.plot(m2[0],m2[1],color2,linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(l2[0],l2[1],color2+'--',label=modelName + '_95%')
    plt.plot(u2[0],u2[1],color2+'--',label=modelName + '_95%')

    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})

    plt.xlabel(xLabel)
    plt.xlim(-0.05,.3)
    plt.ylabel(ylabel)
    plt.ylim(limits[0],limits[1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap' + str(saveName))
    plt.show()

#plotData((timeData,bootstrapControlData['x']),(timeData,bootstrapControlData['x_01']),(timeData,bootstrapControlData['x_99']) \
#,(timeObs,bootstrapControlObs['x']),(timeObs,bootstrapControlObs['x_01']),(timeObs,bootstrapControlObs['x_99']) \
#,style['k'], style['c'], 2, 't (s)','x (cm)',(yLimits['x'][0],yLimits['x'][1]), 'Bootstrap-' + modelName + '-c_o-t_x')

'''

if __name__ == "__main__":
    timeData = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsControl['SampleID'].unique() if sampleID >= -25]
    timeObs = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsControl['SampleID'].unique() if sampleID >= -25]

    plt.clf()
    plt.figure(1,figsize=(12,8))
    dataLegend, = plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
    obsLegend, = plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y'],'b',linewidth=2.0,label=modelName + '_Mean')
    plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x_01'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y_01'],'k--',label='Roach_95%')
    dataLegendS, = plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x_99'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y_99'],'k--',label='Roach_95%')
    obsLegendS, = plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x_01'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y_01'],'b--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x_99'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y_99'],'b--',label=modelName + '_95%')
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
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-x_y')
    plt.show()


    plt.clf()
    plt.figure(2,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['x'],'k',linewidth=2.0,label=modelName + '_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['x_01'],'k--',label=modelName + '_95%')
    plt.plot(timeData,bootstrapDataStatsControl['x_99'],'k--',label=modelName + '_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['x'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['x_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['x_99'],'b--',label=modelName + '_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('x (cm)')
    plt.ylim(yLimits['x'][0],yLimits['x'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_x')
    plt.show()

    plt.clf()
    plt.figure(3,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['y'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['y_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['y_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['y'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['y_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['y_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('y (cm/s)')
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs y', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_y')
    plt.show()

    plt.clf()
    plt.figure(4,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['theta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['theta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['theta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['theta'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['theta_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['theta_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('theta (rad)')
    plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_theta')
    plt.show()

    plt.clf()
    plt.figure(5,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['v'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['v_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['v_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['v'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['v_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['v_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('v (cm/s)')
    plt.ylim(yLimits['v'][0],yLimits['v'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs v', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_v')
    plt.show()

    plt.clf()
    plt.figure(6,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['delta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['delta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['delta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['delta'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['delta_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['delta_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('delta (rad)')
    plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_delta')
    plt.show()

    plt.clf()
    plt.figure(7,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['omega'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['omega_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['omega_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['omega'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['omega_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['omega_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('omega (rad/s)')
    plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_omega')
    plt.show()

    plt.clf()
    plt.figure(8,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['dx'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['dx_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['dx_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['dx'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['dx_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['dx_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dx (cm/s)')
    plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_dx')
    plt.show()

    plt.clf()
    plt.figure(9,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['dy'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['dy_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsControl['dy_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['dy'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['dy_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['dy_99'],'b--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dy (cm/s)')
    plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_dy')
    plt.show()

    plt.clf()
    plt.figure(10,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsControl['acc'],'m',linewidth=2.0,label='Cart_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsControl['acc_01'],'m--',label='Cart_95%')
    plt.plot(timeData,bootstrapDataStatsControl['acc_99'],'m--',label='Cart_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsControl['acc'],'y',linewidth=2.0,label='Model_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsControl['acc_01'],'y--',label='Model_95%')
    plt.plot(timeObs,bootstrapObsStatsControl['acc_99'],'y--',label='Model_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('acc (cm2/s2)')
    plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
    #plt.title(str(modelName) + ' with Control Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Boostrap-'+modelName+'-c_o-t_acc')
    plt.show()

    timeData = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsMass['SampleID'].unique() if sampleID >= -25]
    timeObs = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsMass['SampleID'].unique() if sampleID >= -25]

    plt.clf()
    plt.figure(11,figsize=(12,8))
    dataLegend, = plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
    obsLegend, = plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y'],'r',linewidth=2.0,label=modelName + '_Mean')
    dataLegendS, = plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x_01'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y_01'],'k--',label='Roach_95%')
    plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x_99'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y_99'],'k--',label='Roach_95%')
    obsLegendS, = plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x_01'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y_01'],'r--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x_99'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.xlim(yLimits['x'][0],yLimits['x'][1])
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-x_y')
    plt.show()

    plt.clf()
    plt.figure(12,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['x'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['x_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['x_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['x'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['x_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['x_99'],'r--',label=modelName + '_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('x (cm)')
    plt.ylim(yLimits['x'][0],yLimits['x'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_x')
    plt.show()

    plt.clf()
    plt.figure(13,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['y'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['y_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['y_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['y'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['y_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['y_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('y (cm)')
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs y', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_y')
    plt.show()

    plt.clf()
    plt.figure(14,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['theta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['theta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['theta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['theta'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['theta_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['theta_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('theta (rad)')
    plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_theta')
    plt.show()

    plt.clf()
    plt.figure(15,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['v'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['v_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['v_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['v'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['v_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['v_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('v (cm/s)')
    plt.ylim(yLimits['v'][0],yLimits['v'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs v', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_v')
    plt.show()

    plt.clf()
    plt.figure(16,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['delta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['delta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['delta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['delta'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['delta_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['delta_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('delta (rad)')
    plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_delta')
    plt.show()

    plt.clf()
    plt.figure(17,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['omega'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['omega_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['omega_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['omega'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['omega_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['omega_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('omega (rad/s)')
    plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_omega')
    plt.show()

    plt.clf()
    plt.figure(18,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['dx'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['dx_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['dx_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['dx'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['dx_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['dx_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dx (cm/s)')
    plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_dx')
    plt.show()

    plt.clf()
    plt.figure(19,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['dy'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['dy_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsMass['dy_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['dy'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['dy_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['dy_99'],'r--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dy (cm/s)')
    plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_dy')
    plt.show()

    plt.clf()
    plt.figure(20,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsMass['acc'],'m',linewidth=2.0,label='Cart_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsMass['acc_01'],'m--',label='Cart_95%')
    plt.plot(timeData,bootstrapDataStatsMass['acc_99'],'m--',label='Cart_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsMass['acc'],'y',linewidth=2.0,label='Model_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsMass['acc_01'],'y--',label='Model_95%')
    plt.plot(timeObs,bootstrapObsStatsMass['acc_99'],'y--',label='Model_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('acc (cm2/s2)')
    plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
    #plt.title(str(modelName) + ' with Mass Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-m_o-t_acc')
    plt.show()

    timeData = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsInertia['SampleID'].unique() if sampleID >= -25]
    timeObs = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsInertia['SampleID'].unique() if sampleID >= -25]

    plt.clf()
    plt.figure(21,figsize=(12,8))
    dataLegend, = plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y'],'k',linewidth=2.0,label='Roach_Mean')
    obsLegend, = plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label=modelName + '_Mean')
    dataLegendS, = plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x_01'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y_01'],'k--',label='Roach_95%')
    plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x_99'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y_99'],'k--',label='Roach_95%')
    obsLegendS, = plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x_01'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y_01'],'g--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x_99'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y_99'],'g--',label=modelName + '_95%')
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
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-x_y')
    plt.show()

    plt.clf()
    plt.figure(22,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['x'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['x_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['x_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['x'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['x_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['x_99'],'g--',label=modelName + '_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('x (cm)')
    plt.ylim(yLimits['x'][0],yLimits['x'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_x')
    plt.show()

    plt.clf()
    plt.figure(23,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['y'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['y_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['y_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['y'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['y_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['y_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('y (cm)')
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_y')
    plt.show()

    plt.clf()
    plt.figure(24,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['theta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['theta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['theta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['theta'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['theta_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['theta_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('theta (rad)')
    plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_theta')
    plt.show()

    plt.clf()
    plt.figure(25,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['v'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['v_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['v_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['v'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['v_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['v_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('v (cm/s)')
    plt.ylim(yLimits['v'][0],yLimits['v'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_v')
    plt.show()

    plt.clf()
    plt.figure(26,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['delta'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['delta_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['delta_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['delta'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['delta_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['delta_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('delta (rad)')
    plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_delta')
    plt.show()

    plt.clf()
    plt.figure(27,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['omega'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['omega_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['omega_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['omega'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['omega_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['omega_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('omega (rad/s)')
    plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_omega')
    plt.show()

    plt.clf()
    plt.figure(28,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['dx'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['dx_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['dx_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['dx'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['dx_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['dx_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dx (cm/s)')
    plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_dx')
    plt.show()

    plt.clf()
    plt.figure(29,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['dy'],'k',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['dy_01'],'k--',label='Roach_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['dy_99'],'k--',label='Roach_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['dy'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['dy_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['dy_99'],'g--',label=modelName + '_95%')
    #plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dy (cm/s)')
    plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_dy')
    plt.show()

    plt.clf()
    plt.figure(30,figsize=(12,8))
    dataLegend, = plt.plot(timeData,bootstrapDataStatsInertia['acc'],'m',linewidth=2.0,label='Cart_Mean')
    dataLegendS, = plt.plot(timeData,bootstrapDataStatsInertia['acc_01'],'m--',label='Cart_95%')
    plt.plot(timeData,bootstrapDataStatsInertia['acc_99'],'m--',label='Cart_95%')
    obsLegend, = plt.plot(timeObs,bootstrapObsStatsInertia['acc'],'y',linewidth=2.0,label='Model_Mean')
    obsLegendS, = plt.plot(timeObs,bootstrapObsStatsInertia['acc_01'],'y--',label='Model_95%')
    plt.plot(timeObs,bootstrapObsStatsInertia['acc_99'],'y--',label='Model_95%')
    plt.legend(handles=[dataLegend, obsLegend, dataLegendS, obsLegendS], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('acc (cm2/s2)')
    plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs acc', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-i_o-t_acc')
    plt.show()


    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################

    timeObsControl = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsControl['SampleID'].unique() if sampleID >= -25]

    timeObsMass = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsMass['SampleID'].unique() if sampleID >= -25]

    timeObsInertia = [round(sampleID * .002,4) for sampleID in bootstrapObsStatsInertia['SampleID'].unique() if sampleID >= -25]


    plt.clf()
    plt.figure(31,figsize=(12,8))
    obsLegend, = plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x_01'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y_01'],'b--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['x_99'],bootstrapObsStatsControl[bootstrapObsStatsControl.index <= 150]['y_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x_01'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y_01'],'r--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['x_99'],bootstrapObsStatsMass[bootstrapObsStatsMass.index <= 150]['y_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x_01'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y_01'],'g--',label=modelName + '_95%')
    plt.plot(bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['x_99'],bootstrapObsStatsInertia[bootstrapObsStatsInertia.index <= 150]['y_99'],'g--',label=modelName + '_95%')

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
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-x_y')
    plt.show()

    plt.clf()
    plt.figure(32,figsize=(12,8))

    obsLegendC, = plt.plot(timeObsControl,bootstrapObsStatsControl['x'],'b',linewidth=2.0,label=modelName + '_Control_Mean')
    obsLegendSC, = plt.plot(timeObsControl,bootstrapObsStatsControl['x_01'],'b--',label=modelName + '_Control_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['x_99'],'b--',label=modelName + '_Control_95%')

    obsLegendM, = plt.plot(timeObsMass,bootstrapObsStatsMass['x'],'r',linewidth=2.0,label=modelName + '_Mass_Mean')
    obsLegendSM, = plt.plot(timeObsMass,bootstrapObsStatsMass['x_01'],'r--',label=modelName + '_Mass_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['x_99'],'r--',label=modelName + '_Mass_95%')

    obsLegendI, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['x'],'g',linewidth=2.0,label=modelName + '_Inertia_Mean')
    obsLegendSI, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['x_01'],'g--',label=modelName + '_Inertia_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['x_99'],'g--',label=modelName + '_Inertia_95%')

    plt.legend(handles=[obsLegendC, obsLegendSC, obsLegendM, obsLegendSM, obsLegendI, obsLegendSI], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('x (cm)')
    plt.ylim(yLimits['x'][0],yLimits['x'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_x')
    plt.show()

    plt.clf()
    plt.figure(33,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['y'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['y_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['y_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['y'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['y_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['y_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['y'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['y_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['y_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('y (cm)')
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_y')
    plt.show()

    plt.clf()
    plt.figure(34,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['theta'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['theta_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['theta_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['theta'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['theta_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['theta_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['theta'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['theta_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['theta_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('theta (rad)')
    plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_theta')
    plt.show()

    plt.clf()
    plt.figure(35,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['v'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['v_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['v_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['v'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['v_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['v_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['v'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['v_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['v_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('v (cm/s)')
    plt.ylim(yLimits['v'][0],yLimits['v'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_v')
    plt.show()

    plt.clf()
    plt.figure(36,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['delta'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['delta_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['delta_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['delta'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['delta_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['delta_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['delta'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['delta_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['delta_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('delta (rad)')
    plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_delta')
    plt.show()

    plt.clf()
    plt.figure(37,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['omega'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['omega_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['omega_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['omega'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['omega_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['omega_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['omega'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['omega_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['omega_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('omega (rad/s)')
    plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_omega')
    plt.show()

    plt.clf()
    plt.figure(37,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['dx'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['dx_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['dx_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['dx'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['dx_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['dx_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['dx'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['dx_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['dx_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dx (cm/s)')
    plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_dx')
    plt.show()

    plt.clf()
    plt.figure(39,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['dy'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['dy_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['dy_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['dy'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['dy_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['dy_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['dy'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['dy_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['dy_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dy (cm/s)')
    plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-'+modelName+'-c_m_i-t_dy')
    plt.show()

    plt.clf()
    plt.figure(40,figsize=(12,8))

    obsLegend, = plt.plot(timeObsControl,bootstrapObsStatsControl['acc'],'b',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsControl,bootstrapObsStatsControl['acc_01'],'b--',label=modelName + '_95%')
    plt.plot(timeObsControl,bootstrapObsStatsControl['acc_99'],'b--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsMass,bootstrapObsStatsMass['acc'],'r',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsMass,bootstrapObsStatsMass['acc_01'],'r--',label=modelName + '_95%')
    plt.plot(timeObsMass,bootstrapObsStatsMass['acc_99'],'r--',label=modelName + '_95%')

    obsLegend, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['acc'],'g',linewidth=2.0,label=modelName + '_Mean')
    obsLegendS, = plt.plot(timeObsInertia,bootstrapObsStatsInertia['dy_01'],'g--',label=modelName + '_95%')
    plt.plot(timeObsInertia,bootstrapObsStatsInertia['acc_99'],'g--',label=modelName + '_95%')

    #plt.legend(handles=[obsLegend, obsLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
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


    timeDataControl = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsControl['SampleID'].unique() if sampleID >= -25]

    timeDataMass = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsMass['SampleID'].unique() if sampleID >= -25]

    timeDataInertia = [round(sampleID * .002,4) for sampleID in bootstrapDataStatsInertia['SampleID'].unique() if sampleID >= -25]


    plt.clf()
    plt.figure(41,figsize=(12,8))
    dataLegend, = plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x_01'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y_01'],'b--',label='Roach_95%')
    plt.plot(bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['x_99'],bootstrapDataStatsControl[bootstrapDataStatsControl.index <= 150]['y_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x_01'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y_01'],'r--',label='Roach_95%')
    plt.plot(bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['x_99'],bootstrapDataStatsMass[bootstrapDataStatsMass.index <= 150]['y_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x_01'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y_01'],'g--',label='Roach_95%')
    plt.plot(bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['x_99'],bootstrapDataStatsInertia[bootstrapDataStatsInertia.index <= 150]['y_99'],'g--',label='Roach_95%')

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
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-x_y')
    plt.show()

    plt.clf()
    plt.figure(42,figsize=(12,8))

    dataLegendC, = plt.plot(timeDataControl,bootstrapDataStatsControl['x'],'b',linewidth=2.0,label='Roach_Control_Mean')
    dataLegendSC, = plt.plot(timeDataControl,bootstrapDataStatsControl['x_01'],'b--',label='Roach_Control_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['x_99'],'b--',label='Roach_95%')

    dataLegendM, = plt.plot(timeDataMass,bootstrapDataStatsMass['x'],'r',linewidth=2.0,label='Roach_Mass_Mean')
    dataLegendSM, = plt.plot(timeDataMass,bootstrapDataStatsMass['x_01'],'r--',label='Roach_Mass_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['x_99'],'r--',label='Roach_95%')

    dataLegendI, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['x'],'g',linewidth=2.0,label='Roach_Inertia_Mean')
    dataLegendSI, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['x_01'],'g--',label='Roach_Inertia_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['x_99'],'g--',label='Roach_95%')

    plt.legend(handles=[dataLegendC, dataLegendSC, dataLegendM, dataLegendSM, dataLegendI, dataLegendSI], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('x (cm)')
    plt.ylim(yLimits['x'][0],yLimits['x'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_x')
    plt.show()

    plt.clf()
    plt.figure(43,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['y'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['y_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['y_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['y'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['y_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['y_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['y'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['y_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['y_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('y (cm)')
    plt.ylim(yLimits['y'][0],yLimits['y'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs y', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_y')
    plt.show()

    plt.clf()
    plt.figure(44,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['theta'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['theta_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['theta_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['theta'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['theta_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['theta_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['theta'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['theta_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['theta_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('theta (rad)')
    plt.ylim(yLimits['theta'][0],yLimits['theta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs theta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_theta')
    plt.show()

    plt.clf()
    plt.figure(45,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['v'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['v_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['v_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['v'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['v_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['v_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['v'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['v_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['v_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('v (cm/s)')
    plt.ylim(yLimits['v'][0],yLimits['v'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs v', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_v')
    plt.show()

    plt.clf()
    plt.figure(46,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['delta'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['delta_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['delta_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['delta'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['delta_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['delta_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['delta'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['delta_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['delta_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('delta (rad)')
    plt.ylim(yLimits['delta'][0],yLimits['delta'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs delta', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_delta')
    plt.show()

    plt.clf()
    plt.figure(47,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['omega'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['omega_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['omega_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['omega'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['omega_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['omega_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['omega'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['omega_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['omega_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('omega (rad/s)')
    plt.ylim(yLimits['omega'][0],yLimits['omega'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs omega', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_omega')
    plt.show()

    plt.clf()
    plt.figure(48,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['dx'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['dx_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['dx_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['dx'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['dx_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['dx_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['dx'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['dx_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['dx_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dx (cm/s)')
    plt.ylim(yLimits['dx'][0],yLimits['dx'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dx', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_dx')
    plt.show()

    plt.clf()
    plt.figure(49,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['dy'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['dy_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['dy_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['dy'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['dy_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['dy_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['dy'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['dy_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['dy_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('dy (cm/s)')
    plt.ylim(yLimits['dy'][0],yLimits['dy'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_dy')
    plt.show()

    plt.clf()
    plt.figure(50,figsize=(12,8))

    dataLegend, = plt.plot(timeDataControl,bootstrapDataStatsControl['acc'],'b',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataControl,bootstrapDataStatsControl['acc_01'],'b--',label='Roach_95%')
    plt.plot(timeDataControl,bootstrapDataStatsControl['acc_99'],'b--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataMass,bootstrapDataStatsMass['acc'],'r',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataMass,bootstrapDataStatsMass['acc_01'],'r--',label='Roach_95%')
    plt.plot(timeDataMass,bootstrapDataStatsMass['acc_99'],'r--',label='Roach_95%')

    dataLegend, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['acc'],'g',linewidth=2.0,label='Roach_Mean')
    dataLegendS, = plt.plot(timeDataInertia,bootstrapDataStatsInertia['acc_01'],'g--',label='Roach_95%')
    plt.plot(timeDataInertia,bootstrapDataStatsInertia['acc_99'],'g--',label='Roach_95%')

    #plt.legend(handles=[dataLegend, dataLegendS], bbox_to_anchor=(1.35, 1), loc=1, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('acc (cm/s)')
    plt.ylim(yLimits['acc'][0],yLimits['acc'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs dy', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_acc')
    plt.show()

    plt.clf()
    plt.figure(52,figsize=(12,8))

    dataLegendC, = plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['x'])),'b',linewidth=2.0,label='Roach_Control_Mean')
    dataLegendSC, = plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['x_01'])),'b--',label='Roach_Control_95%')
    plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['x_99'])),'b--',label='Roach_95%')

    dataLegendM, = plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['x'])),'r',linewidth=2.0,label='Roach_Mass_Mean')
    dataLegendSM, = plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['x_01'])),'r--',label='Roach_Mass_95%')
    plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['x_99'])),'r--',label='Roach_95%')

    dataLegendI, = plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['x'])),'g',linewidth=2.0,label='Roach_Inertia_Mean')
    dataLegendSI, = plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['x_01'])),'g--',label='Roach_Inertia_95%')
    plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['x_99'])),'g--',label='Roach_95%')

    plt.legend(handles=[dataLegendC, dataLegendSC, dataLegendM, dataLegendSM, dataLegendI, dataLegendSI], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('roach x acc (cm/s^2)')
    plt.ylim(yLimits['accX'][0],yLimits['accX'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_roachxacc')
    plt.show()

    plt.clf()
    plt.figure(52,figsize=(12,8))

    dataLegendC, = plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['y'])),'b',linewidth=2.0,label='Roach_Control_Mean')
    dataLegendSC, = plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['y_01'])),'b--',label='Roach_Control_95%')
    plt.plot(timeDataControl[1:len(timeDataControl)-1],np.diff(np.diff(bootstrapDataStatsControl['y_99'])),'b--',label='Roach_95%')

    dataLegendM, = plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['y'])),'r',linewidth=2.0,label='Roach_Mass_Mean')
    dataLegendSM, = plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['y_01'])),'r--',label='Roach_Mass_95%')
    plt.plot(timeDataMass[1:len(timeDataMass)-1],np.diff(np.diff(bootstrapDataStatsMass['y_99'])),'r--',label='Roach_95%')

    dataLegendI, = plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['y'])),'g',linewidth=2.0,label='Roach_Inertia_Mean')
    dataLegendSI, = plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['y_01'])),'g--',label='Roach_Inertia_95%')
    plt.plot(timeDataInertia[1:len(timeDataInertia)-1],np.diff(np.diff(bootstrapDataStatsInertia['y_99'])),'g--',label='Roach_95%')

    plt.legend(handles=[dataLegendC, dataLegendSC, dataLegendM, dataLegendSM, dataLegendI, dataLegendSI], loc=2, prop={'size':12})
    plt.xlabel('t (s)')
    plt.xlim(-0.05,.3)
    plt.ylabel('roach y acc (cm/s^2)')
    plt.ylim(yLimits['accY'][0],yLimits['accY'][1])
    #plt.title(str(modelName) + ' with Inertia Treatment and ' + str(treatment) + ': t vs x', y=1.08)
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('RobertFull/Bootstrap-c_m_i_d-t_roachyacc')
    plt.show()
