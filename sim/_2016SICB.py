import modelwrapper as model
import modelplot as modelplot
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.patches as mpatches
import os
import time
import numpy as np
import copy
import random
import kalman
import pandas as pd
import pwhamil as pw
import scipy.optimize as spo
from shrevz import util as shutil
import lls


def pandasToNumpy(obs, minIndex, maxIndex):
    return np.asarray(obs['x'][minIndex:maxIndex+1]), \
    np.asarray(obs['y'][minIndex:maxIndex+1]), \
    np.asarray(obs['theta'][minIndex:maxIndex+1]), \
    np.asarray(obs['accx'][minIndex:maxIndex+1]), \
    np.asarray(obs['accy'][minIndex:maxIndex+1]), \
    np.asarray(obs['acctheta'][minIndex:maxIndex+1])

def centerAtOrigin(x, y):
    xOrigin = x[0]
    yOrigin = y[0]
    return x - xOrigin, y - yOrigin

def findAverageHeading(x, y, theta):
    xyF = np.matrix([x, y])
    U, s, V = np.linalg.svd(xyF)
    thetaAvg = np.arctan2(U.item(1,0), U.item(0,0))
    if abs(thetaAvg) > np.pi/2:
        thetaAvg = np.arctan2(-1.0 * U.item(1,0), -1.0 * U.item(0,0))
    return thetaAvg

def aroundX(x, y, theta, ddx, ddy):
    angle = np.arctan2(y[-1], x[-1])
    X =  np.dot(mw.rotMat2(angle), np.vstack((x,y)))
    DDX =  np.dot(mw.rotMat2(angle), np.vstack((ddx,ddy)))
    theta -= angle
    return X[0,:], X[1,:], theta, DDX[0,:], DDX[1,:]

def downSample(x, y, theta, ddx, ddy, ddtheta):
    axis = np.linspace(0, 2*np.pi, np.shape(x)[0])
    downAxis = np.linspace(0, 2*np.pi, 64)
    x = np.interp(downAxis, axis, x.T)
    y = np.interp(downAxis, axis, y)
    theta = np.interp(downAxis, axis, theta)
    ddx = np.interp(downAxis, axis, ddx)
    ddy = np.interp(downAxis, axis, ddy)
    ddtheta = np.interp(downAxis, axis, ddtheta)
    return downAxis, x, y, theta, ddx, ddy, ddtheta

def presentationLLS(obsID, template, minIndex, maxIndex):
    kalman.modelAccel(mw.observations[obsID], template)
    x, y, theta, ddx, ddy, ddtheta = pandasToNumpy(mw.observations[obsID],minIndex,maxIndex)
    x, y = centerAtOrigin(x, y)
    x, y, theta, ddx, ddy = aroundX(x, y, theta, ddx, ddy)
    phaseAxis, x, y, theta, ddx, ddy, ddtheta = downSample(x, y, theta, ddx, ddy, ddtheta)
    results = pd.DataFrame(index=range(64),columns=['phase', 'x', 'y', 'theta', 'ddx', 'ddy', 'ddtheta'])
    results['phase'] = phaseAxis
    results['x'] = x
    results['y'] = y
    results['theta'] = theta
    results['ddx'] = ddx
    results['ddy'] = ddy
    results['ddtheta'] = ddtheta
    return results

def dataPlot(pwTable, llsTable, estimatesLabel, label):
    rescaledAxis =  np.linspace(0, 100, len(llsTable['phase']))
    G_CONVERSION = 0.00101971621
    estimatesTable = pw.loadEstimatesTable(estimatesLabel)

    (posX, _, accelX) = kalman.columnTableX('Roach_x')
    plt.figure(1,figsize=(12,8)); plt.clf(); plt.hold(True);
    dataLegend, = plt.plot(rescaledAxis, estimatesTable[accelX + '_Mean'] * G_CONVERSION, 'k-', lw=3., markersize=12, label='Data', zorder='1')
    dataLegendI, = plt.plot(rescaledAxis, estimatesTable[accelX + '_01'] * G_CONVERSION, 'k-', lw=1., markersize=12, label='98% CI', zorder='1')
    plt.plot(rescaledAxis, estimatesTable[accelX + '_99'] * G_CONVERSION, 'k-', lw=1., markersize=12, label='kalman Estimate', zorder='1')
    plt.fill_between(rescaledAxis, estimatesTable[accelX + '_01'].astype(float) * G_CONVERSION, \
    estimatesTable[accelX + '_99'].astype(float) * G_CONVERSION, facecolor='#D3D3D3', alpha=1.0, zorder='0')
    plt.xlabel('phase (percent)', fontsize=20)
    plt.xlim(0,100)
    plt.ylabel('$\ddot{x}$ (g)', fontsize=24)
    #plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    ax.tick_params(axis='both', labelsize=20)
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    lgd = plt.legend(handles=[dataLegend, dataLegendI], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0, prop={'size':22})
    plt.savefig('2016SICB/plots/d-' + label + '-ddx.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    llsLegend, = plt.plot(rescaledAxis, llsTable['ddx'] * G_CONVERSION, 'b--', lw=3., markersize=12, label='LLS', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., prop={'size':22})
    plt.savefig('2016SICB/plots/dl-' + label + '-ddx.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    patches = [mpatches.Rectangle([50, -.3], 50.0, .7, zorder=-5, facecolor='#F8F8FF', alpha=0.3, hatch='//')]
    ax.add_patch(patches[0])
    pwhamilLegend, = plt.plot(rescaledAxis[0:32], pwTable['DDX'][0:32] * G_CONVERSION, 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    pwhamilLegend, = plt.plot(rescaledAxis[32:64], pwTable['DDX'][32:64] * G_CONVERSION, 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend, pwhamilLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, borderaxespad=0., prop={'size':18})
    plt.savefig('2016SICB/plots/dlp-' + label + '-ddx.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    plt.show()

    (posX, _, accelX) = kalman.columnTableX('Roach_y')
    plt.figure(1,figsize=(12,8)); plt.clf(); plt.hold(True);
    dataLegend, = plt.plot(rescaledAxis, estimatesTable[accelX + '_Mean'] * G_CONVERSION, 'k-', lw=3., markersize=12, label='Data', zorder='1')
    dataLegendI, = plt.plot(rescaledAxis, estimatesTable[accelX + '_01'] * G_CONVERSION, 'k-', lw=1., markersize=12, label='98% CI', zorder='1')
    plt.plot(rescaledAxis, estimatesTable[accelX + '_99'] * G_CONVERSION, 'k-', lw=1., markersize=12, label='kalman Estimate', zorder='1')
    plt.fill_between(rescaledAxis, estimatesTable[accelX + '_01'].astype(float) * G_CONVERSION, \
    estimatesTable[accelX + '_99'].astype(float) * G_CONVERSION, facecolor='#D3D3D3', alpha=1.0, zorder='0')
    plt.xlabel('phase (percent)', fontsize=20)
    plt.xlim(0,100)
    plt.ylabel('$\ddot{y}$ (g)', fontsize=24)
    #plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    ax.tick_params(axis='both', labelsize=20)
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    lgd = plt.legend(handles=[dataLegend, dataLegendI], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0, prop={'size':22})
    plt.savefig('2016SICB/plots/d-' + label + '-ddy.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    llsLegend, = plt.plot(rescaledAxis, llsTable['ddy'] * G_CONVERSION, 'b--', lw=3., markersize=12, label='LLS', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., prop={'size':22})
    plt.savefig('2016SICB/plots/dl-' + label + '-ddy.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    patches = [mpatches.Rectangle([50, -.2], 50.0, .45, zorder=-5, facecolor='#F8F8FF', alpha=0.3, hatch='//')]
    ax.add_patch(patches[0])
    pwhamilLegend, = plt.plot(rescaledAxis[0:32], pwTable['DDY'][0:32] * G_CONVERSION, 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    pwhamilLegend, = plt.plot(rescaledAxis[32:64], pwTable['DDY'][32:64] * G_CONVERSION, 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend, pwhamilLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, borderaxespad=0., prop={'size':18})
    plt.savefig('2016SICB/plots/dlp-' + label + '-ddy.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    plt.show()

    (posX, _, accelX) = kalman.columnTableX('Roach_theta')
    plt.figure(1,figsize=(12,8)); plt.clf(); plt.hold(True);
    dataLegend, = plt.plot(rescaledAxis, estimatesTable[accelX + '_Mean'], 'k-', lw=3., markersize=12, label='Data', zorder='1')
    dataLegendI, = plt.plot(rescaledAxis, estimatesTable[accelX + '_01'], 'k-', lw=1., markersize=12, label='98% CI', zorder='1')
    plt.plot(rescaledAxis, estimatesTable[accelX + '_99'], 'k-', lw=1., markersize=12, label='kalman Estimate', zorder='1')
    plt.fill_between(rescaledAxis, estimatesTable[accelX + '_01'].astype(float), \
    estimatesTable[accelX + '_99'].astype(float), facecolor='#D3D3D3', alpha=1.0, zorder='0')
    plt.xlabel('phase (percent)', fontsize=20)
    plt.xlim(0,100)
    plt.ylim(-10,11)
    plt.ylabel('$\ddot{ {\\theta} }$ (rad/$s^2$)', fontsize=24)
    #plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    ax.tick_params(axis='both', labelsize=20)
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    lgd = plt.legend(handles=[dataLegend, dataLegendI], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0, prop={'size':22})
    plt.savefig('2016SICB/plots/d-' + label + '-ddtheta.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    llsLegend, = plt.plot(rescaledAxis, llsTable['ddtheta'], 'b--', lw=3., markersize=12, label='LLS', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., prop={'size':22})
    plt.savefig('2016SICB/plots/dl-' + label + '-ddtheta.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    patches = [mpatches.Rectangle([50, -10], 50.0, 21, zorder=-5, facecolor='#F8F8FF', alpha=0.3, hatch='//')]
    ax.add_patch(patches[0])
    pwhamilLegend, = plt.plot(rescaledAxis[0:32], pwTable['DDTheta'][0:32], 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    pwhamilLegend, = plt.plot(rescaledAxis[32:64], pwTable['DDTheta'][32:64], 'r--', lw=4., markersize=8, label='PWHamil', zorder='1')
    lgd = plt.legend(handles=[dataLegend, dataLegendI, llsLegend, pwhamilLegend], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, borderaxespad=0., prop={'size':18})
    plt.savefig('2016SICB/plots/dlp-' + label + '-ddtheta.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=200)
    plt.show()

def dataPipelinePlot(data, template):
    np.random.seed(7)
    POINTS = 100
    NUM_PERIODS = 4
    PHASE_LENGTH = 2*np.pi*NUM_PERIODS
    AMPLITUDE = 0.15
    ORDER = 4
    axis = np.linspace(0, PHASE_LENGTH, POINTS)
    strideAxis = axis/(2*np.pi)
    percentAxis = axis/(2 * np.pi) * 100.0
    trajectory = AMPLITUDE * np.sin(axis) + np.random.normal(0, 0.05, POINTS)

    kty = kalman.KalmanTuner('tune_x', template, 'y', [template['v']*template['dt']/100, 0.0, 0.0, .0001, 0.0 , 1000.0], 0)
    oy = spo.leastsq(kty.costy, kty.decVar, full_output=True, maxfev=100, diag=[1e3, 1e1, 1e-2, 1e-2, 1e-3, 1e-3])

    (mu, cov, x, P) = kalman.dataKalman(trajectory, kty.kalmanFilter, oy[0])

    fsX = shutil.FourierSeries()
    fsDX = shutil.FourierSeries()
    fsDDX = shutil.FourierSeries()

    fsX.fit(ORDER, axis[np.newaxis], x[:,0][np.newaxis])
    fsDX.fit(ORDER, axis[np.newaxis], x[:,1][np.newaxis])
    fsDDX.fit(ORDER, axis[np.newaxis], x[:,2][np.newaxis])

    x_f = fsX.val(axis[np.newaxis])[:,0].real
    dx_f = fsDX.val(axis[np.newaxis])[:,0].real
    ddx_f = fsDDX.val(axis[np.newaxis])[:,0].real

    fig = plt.figure(1,figsize=(12,8))
    ax = plt.gca()
    plt.plot(strideAxis, trajectory, 'k.-', markersize=10)
    plt.xlabel('strides', fontsize=22)
    plt.ylabel('pos. ($cm$)', fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    plt.xlim(0,NUM_PERIODS)
    plt.ylim(-0.25,0.25)
    plt.savefig('2016SICB/plots/datapipeline-1', bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(2,figsize=(12,20))
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, figsize=(12,8))

    #ax1.set_xlabel('phase (radians)', fontsize=22)
    ax1.set_ylabel('pos. ($cm$)', fontsize=22)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_xlim(0,NUM_PERIODS)
    ax1.set_ylim(-0.25,0.25)
    ax1.plot(strideAxis, x[:,0], 'k.-', markersize=10)

    #ax2.set_xlabel('phase (radians)', fontsize=22)
    ax2.set_ylabel('vel. $(cm/s)$', fontsize=22)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_xlim(0,NUM_PERIODS)
    ax2.set_ylim(-15,15)
    ax2.plot(strideAxis, x[:,1], 'k.-', markersize=10)

    ax3.set_xlabel('strides', fontsize=22)
    ax3.set_ylabel('accel. ($cm/s^2$)', fontsize=22)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.set_xlim(0,NUM_PERIODS)
    ax3.set_ylim(-2500,2500)
    ax3.plot(strideAxis, x[:,2], 'k.-', markersize=10)

    fig.tight_layout()
    plt.savefig('2016SICB/plots/datapipeline-2', bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(3,figsize=(12,20))
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, figsize=(12,8))

    plotSegment = POINTS/NUM_PERIODS
    #ax1.set_xlabel('phase (radians)', fontsize=22)
    ax1.set_ylabel('pos. ($cm)$', fontsize=22)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_xlim(0,100)
    ax1.set_ylim(-0.25,0.25)
    ax1.plot(percentAxis[:plotSegment+3], x_f[:plotSegment+3], 'k--', lw=6.5, zorder='2')
    ax1.plot(percentAxis[:plotSegment+2], x[0:plotSegment+2,0], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax1.plot(percentAxis[:plotSegment+2], x[plotSegment:2*plotSegment+2,0], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax1.plot(percentAxis[:plotSegment+2], x[2*plotSegment:3*plotSegment+2,0], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax1.plot(percentAxis[:plotSegment+2], x[3*plotSegment-2:4*plotSegment,0], 'k.-', markersize=10, zorder='1', alpha=0.3)

    #ax2.set_xlabel('phase (radians)', fontsize=22)
    ax2.set_ylabel('vel. ($cm/s)$', fontsize=22)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_xlim(0,100)
    ax2.set_ylim(-15,15)
    ax2.plot(percentAxis[:plotSegment+3], dx_f[:plotSegment+3], 'k--', lw=6.5, zorder='2')
    ax2.plot(percentAxis[:plotSegment+2], x[0:plotSegment+2,1], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax2.plot(percentAxis[:plotSegment+2], x[plotSegment:2*plotSegment+2,1], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax2.plot(percentAxis[:plotSegment+2], x[2*plotSegment:3*plotSegment+2,1], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax2.plot(percentAxis[:plotSegment+2], x[3*plotSegment-2:4*plotSegment+2,1], 'k.-', markersize=10, zorder='1', alpha=0.3)


    ax3.set_xlabel('phase (percent)', fontsize=22)
    ax3.set_ylabel('accel. ($cm/s^2)$', fontsize=22)
    ax3.tick_params(axis='both', labelsize=20)
    ax3.set_xlim(0,100)
    ax3.set_ylim(-2500,2500)
    ax3.plot(percentAxis[:plotSegment+3], ddx_f[:plotSegment+3], 'k--', lw=6.5, zorder='2')
    ax3.plot(percentAxis[:plotSegment+2], x[0:plotSegment+2,2], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax3.plot(percentAxis[:plotSegment+2], x[plotSegment:2*plotSegment+2,2], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax3.plot(percentAxis[:plotSegment+2], x[2*plotSegment:3*plotSegment+2,2], 'k.-', markersize=10, zorder='1', alpha=0.3)
    ax3.plot(percentAxis[:plotSegment+2], x[3*plotSegment-2:4*plotSegment,2], 'k.-', markersize=10, zorder='1', alpha=0.3)

    fig.tight_layout()
    plt.savefig('2016SICB/plots/datapipeline-3', bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(4,figsize=(12,20))
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, figsize=(12,8))
    #ax1.set_xlabel('phase (radians)', fontsize=22)
    ax1.set_ylabel('pos. ($cm$)', fontsize=22)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_xlim(0,100)
    ax1.set_ylim(-0.25,0.25)
    ax1.plot(percentAxis[:plotSegment+3], x_f[:plotSegment+3], 'k--', lw=6.5, zorder='2')


    for i in range(1,2):
        ax1.plot((50,50),(-0.25,0.25), 'r--', linewidth=5, zorder='1')

    #for i in [0]:
        #ax1.fill_between((np.pi*i,np.pi*(i+1)), -0.25, 0.25, facecolor='#8A2BE2', alpha=0.25, zorder='0')

    patches = [mpatches.Rectangle([50, -0.25], 50, 50, zorder=-5, facecolor='#F8F8FF', alpha=0.3, hatch='//')]
    ax1.add_patch(patches[0])


    #for i in [1,3,5,7]:
    #    ax1.fill_between((np.pi*i,np.pi*(i+1)), -0.25, 0.25, facecolor='#FFFF66 ', alpha=0.25, zorder='0')

    ax2.set_xlabel('period', fontsize=22)
    ax2.set_ylabel('accel. ($cm/s^2$)', fontsize=22)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_xlim(0,100)
    ax2.set_ylim(-2000,2000)
    ax2.plot(percentAxis[:plotSegment+3], ddx_f[:plotSegment+3], 'k--', lw=6.5)

    for i in range(1,2):
        ax2.plot((50,50),(-2000,2000), 'r--', linewidth=5)

    #for i in [0]:
        #ax2.fill_between((np.pi*i,np.pi*(i+1)), -2000, 2000, facecolor='#8A2BE2', alpha=0.25, zorder='0')

    patches = [mpatches.Rectangle([50, -2000], 50, 4000, zorder=-5, facecolor='#F8F8FF', alpha=0.3, hatch='//')]
    ax2.add_patch(patches[0])

    fig.tight_layout()
    plt.savefig('2016SICB/plots/datapipeline-4', bbox_inches='tight', dpi=200)
    plt.show()

def LLSAnimation(template, data, observation):
    a = lls.LLS(mc.packConfiguration(template))
    #SteadyState
    #a.anim(mw.observations[0][0:125], saveDir=os.path.join('2016SICB','animations'), label='ss-')
    #Perturbation
    a.animAccel(250,observation[250:311], saveDir=os.path.join('2016SICB','animations'), label='p-', accel=data['CartAcceleration'][250:311])

if __name__ == "__main__":
    saveDir = '2016SICB'

    #index0 and index1 are tuples

    varList = ['x','y','theta','fx','fy','dtheta','omega','q','v','delta','t', 'dx', 'dy', 'hx', 'hy']
    #varList = ['t','x','y','theta','dx','dy','fx','fy','dtheta','delta','v','q']

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)

    mw.csvLoadData([0, 1, 2])
    mw.csvLoadObs([0, 1, 2])
    template = mc.jsonLoadTemplate("templateControl")

    #Animations
    #Test - Perturbation
    #LLSAnimation(copy.deepcopy(template), mw.data[1], mw.observations[1])
    #StableOrbit
    #LLSAnimation(copy.deepcopy(template), mw.data[0], mw.observations[0])

    #DataPipeline
    dataPipelinePlot(mw.data[0], template)

    #Generate x,y,theta plots
    #Stride indices not offset!!!!
    #control: 14092, 14472
    #mass: 14037, 14415
    #inertia: 14057, 14436

    '''
    template = mc.jsonLoadTemplate("templateControl")
    llsControl = presentationLLS(0, template, 14092, 14472)
    estimatesTable = pw.estimatesTransform('control-estimates')
    results = pw.estimatesTableStanceInfo(estimatesTable, 'control-estimates')
    pw.runPWHamilEstimates('control-estimates', 'control-estimates')
    pwControl = pw.estimatesGeneratePWHamilTable(estimatesTable, 'control-estimates')
    dataPlot(pwControl, llsControl, 'control-estimates', 'test')

    template = mc.jsonLoadTemplate("templateMass")
    llsMass = presentationLLS(1, template, 14037, 14415)
    estimatesTable = pw.estimatesTransform('mass-estimates')
    results = pw.estimatesTableStanceInfo(estimatesTable, 'mass-estimates')
    pw.runPWHamilEstimates('mass-estimates', 'mass-estimates')
    pwMass = pw.estimatesGeneratePWHamilTable(estimatesTable, 'mass-estimates')
    dataPlot(pwMass, llsMass, 'mass-estimates', 'test')

    template = mc.jsonLoadTemplate("templateInertia")
    llsInertia = presentationLLS(2, template, 14057, 14436)
    estimatesTable = pw.estimatesTransform('inertia-estimates')
    results = pw.estimatesTableStanceInfo(estimatesTable, 'inertia-estimates')
    pw.runPWHamilEstimates('inertia-estimates', 'inertia-estimates')
    pwInertia = pw.estimatesGeneratePWHamilTable(estimatesTable, 'inertia-estimates')
    dataPlot(pwInertia, llsInertia, 'inertia-estimates', 'test')
    '''

    '''
    mw.csvLoadData([0,1,2])
    template = mc.jsonLoadTemplate("templateInertia")
    kalman.modelAccel(mw.observations[1], template)
    template = mc.jsonLoadTemplate("templateMass")
    kalman.modelAccel(mw.observations[2], template)

    template = mc.jsonLoadTemplate("templateControl")
    template['dt'] = .0002
    mw.runTrial('lls.LLS', template, varList, 0, 'noAccel')
    template = mc.jsonLoadTemplate("templateMass")
    template['dt'] = .0002
    mw.runTrial('lls.LLS', template, varList, 1, 'noAccel')
    template = mc.jsonLoadTemplate("templateInertia")
    template['dt'] = .0002
    mw.runTrial('lls.LLS', template, varList, 2, 'noAccel')
    mw.saveTables()
    '''
