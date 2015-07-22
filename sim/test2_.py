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

LINE_LENGTH = 8

def calcAccelSam(observation, template, i):
    theta = observation.ix[i, 'theta'] - np.pi/2
    x = observation.ix[i, 'x']
    y = observation.ix[i, 'y']
    fx = observation.ix[i, 'fx']
    fy = observation.ix[i, 'fy']

    f = np.array([fx,fy])
    c = np.array([x,y])

    h = c + template['d'] * np.array([np.sin(theta),np.cos(theta)])
    # leg length
    eta = np.linalg.norm(h - f)
    dV = template['k'] * (eta - template['eta0'])

    # Cartesian dynamics
    dx = [observation.ix[i, 'dx'], observation.ix[i, 'dy'], observation.ix[i, 'dtheta'],
        -dV*(x + template['d']*np.sin(theta) - fx)/(template['m']*eta),
        -dV*(y + template['d']*np.cos(theta) - fy)/(template['m']*eta),
        -dV*template['d']*((x - fx)*np.cos(theta)
        - (y - fy)*np.sin(theta))/(template['I']*eta)]

    return (dx[3], dx[4], dx[5])

def calcAccelBora(observation, template, i):
    print 'ya'

def calcBora(observation, template):
    xDiff = np.diff(np.diff(observation['x']))
    yDiff = np.diff(np.diff(observation['y']))
    rotDiff = np.diff(observation['omega'])
    xDiff = list(xDiff)
    yDiff = list(yDiff)
    rotDiff = list(rotDiff)
    xDiff.insert(0,0)
    yDiff.insert(0,0)
    rotDiff.insert(0,0)
    xDiff.append(0)
    yDiff.append(0)
    rotDiff.append(0)

    observation['xAccel'] = 0
    observation['yAccel'] = 0
    #original rotate theta x,y then translate torque
    observation['xForce'] = 0
    observation['yForce'] = 0
    observation['torque'] = 0
    #nothing
    observation['xForceAlt'] = 0
    observation['yForceAlt'] = 0
    #torque and rotate at the same time
    observation['xForceAlt'] = 0
    observation['yForceAlt'] = 0

    for i in observation.index:
        #accelVector = mw.rotMat3(observation.ix[i ,'theta']) * np.matrix([[xDiff[i]], [yDiff[i]], [0]])
        accelVector = np.matrix([[xDiff[i]], [yDiff[i]], [0]])
        observation.ix[i, 'xAccel'] = accelVector.item(0,0)
        observation.ix[i, 'yAccel'] = accelVector.item(1,0)
        observation.ix[i, 'xForce'] = accelVector.item(0,0) * template['m']
        observation.ix[i, 'yForce'] = accelVector.item(1,0) * template['m']
        observation.ix[i, 'torque'] = rotDiff[i] * template['I']

        #fb = np.matrix([[observation.ix[i, 'xForce'],observation.ix[i, 'yForce'],0]])

        pbc = np.matrix([[template['d'] * np.cos(observation.ix[i, 'theta']), template['d'] * np.sin(observation.ix[i, 'theta']), 0]])
        phat = np.matrix([[0, 0, pbc.item(0,1)], [0, 0, pbc.item(0,0)], [pbc.item(0,1), pbc.item(0,0), 0]])

        zeroMatrix = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ft = np.matrix([[observation.ix[i, 'xForce']], [observation.ix[i, 'yForce']], [0], [0], [0], [observation.ix[i, 'torque']]])

        observation.ix[i, 'xForceOrig'] = -xDiff[i]
        observation.ix[i, 'yForceOrig'] = -yDiff[i]

        transformation = np.vstack([np.hstack([mw.rotMat3(observation.ix[i, 'theta']).transpose(), \
        zeroMatrix]),np.hstack([-mw.rotMat3(observation.ix[i, 'theta']).transpose()*phat, mw.rotMat3(observation.ix[i, 'theta']).transpose()])])

        #transformation = np.vstack([np.hstack([mw.rotMat3(0).transpose(), \
        #zeroMatrix]),np.hstack([-mw.rotMat3(0).transpose()*phat, mw.rotMat3(0).transpose()])])

        fc = transformation * ft

        observation.ix[i, 'xForce'] = -fc.item(0,0)
        observation.ix[i, 'yForce'] = -fc.item(1,0)
        observation.ix[i, 'torque'] = -fc.item(5,0)

    return observation

def calcSam(observation, template):

    observation['xAccel'] = 0
    observation['yAccel'] = 0
    #original rotate theta x,y then translate torque
    observation['xForce'] = 0
    observation['yForce'] = 0
    observation['torque'] = 0
    #nothing
    observation['xForceAlt'] = 0
    observation['yForceAlt'] = 0
    #torque and rotate at the same time
    observation['xForceAlt'] = 0
    observation['yForceAlt'] = 0

    for i in observation.index:
        (accelX, accelY, accelAng) = calcAccelSam(observation, template, i)
        observation.ix[i, 'xForceOrig'] = -accelX * template['m']
        observation.ix[i, 'yForceOrig'] = -accelY * template['m']
        observation.ix[i, 'yForceOrig'] = accelAng * template['I']

        zeroMatrix = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ft = np.matrix([[observation.ix[i, 'xForceOrig']], [observation.ix[i, 'yForceOrig']], [0], [0], [0], [accelAng * template['I']]])

        pbc = np.matrix([[template['d'] * np.cos(observation.ix[i, 'theta']), template['d'] * np.sin(observation.ix[i, 'theta']), 0]])
        #pbc = np.matrix([[template['d'], 0, 0]]).transpose()

        phat = np.matrix([[0, 0, pbc.item(0,1)], [0, 0, pbc.item(0,0)], [pbc.item(0,1), pbc.item(0,0), 0]])

        #transformation = np.vstack([np.hstack([mw.rotMat3(observation.ix[i, 'theta']).transpose(), \
        #zeroMatrix]),np.hstack([-mw.rotMat3(observation.ix[i, 'theta']).transpose()*phat, mw.rotMat3(observation.ix[i, 'theta']).transpose()])])

        transformation = np.vstack([np.hstack([mw.rotMat3(0).transpose(), \
        zeroMatrix]),np.hstack([-mw.rotMat3(0).transpose()*phat, mw.rotMat3(0).transpose()])])

        fc = transformation * ft

        observation.ix[i, 'xForce'] = fc.item(0,0)
        observation.ix[i, 'yForce'] = fc.item(1,0)
        observation.ix[i, 'torque'] = fc.item(5,0)

    return observation

def plotLines(observation, figNum, xFstring, yFstring):
    plt.figure(figNum, figsize=(10,10)); plt.clf()
    for i in range(100):
        x = observation.ix[i, 'x']
        y = observation.ix[i, 'y']
        xF = observation.ix[i, xFstring]
        yF = observation.ix[i, yFstring]
        alpha = np.sqrt(LINE_LENGTH/(xF**2 + yF**2))
        plt.plot([x - template['d'] * np.cos(observation.ix[i, 'theta']), x + xF * alpha - template['d'] * np.cos(observation.ix[i, 'theta'])], \
         [y - template['d'] * np.sin(observation.ix[i, 'theta']), y + yF * alpha - template['d'] * np.sin(observation.ix[i, 'theta'])], color='b', linestyle='-', linewidth=1)
        plt.plot(x + template['d'] * np.sin(observation.ix[i, 'theta'] - np.pi/2), y + template['d'] * np.cos(observation.ix[i, 'theta'] - np.pi/2), c='g')
        plt.plot(observation.ix[i, 'fx'], observation.ix[i, 'fy'], marker='.', c='k')
        plt.plot(observation.ix[i, 'x'], observation.ix[i, 'y'], marker='.', c='r')
    plt.show()

observation = mw.observations[0]
template = mc.jsonLoadTemplate('templateControl')
#observation = calcBora(observation, template)
observation = calcSam(observation, template)

plotLines(observation, 1, 'xForce', 'yForce')
#plotLines(observation, 2, 'xForceOrig', 'yForceOrig')
