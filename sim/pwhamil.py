import numpy as np
import pylab as plt

from util import poly
import sympy
import string
import footlocation
import modelwrapper as model
import pickle
import os
import kalman
import pandas as pd
from collections import deque

def fit(q,a,dv,m_=None,m=None):
    """
    c = fit(q,a,dv)  fit potential to accel data using least-squares
    i.e. if a = c dV(q) where dV(q) = sum([dv[j](q) for j in range(len(dv))])
    then c = a^T X (X^T X)^{-1} where X = dV(q)

    Inputs:
        q - N x Nd - N samples of Nd dimensional configurations
        a - N x Nd - N samples of Nd dimensional accelerations
        dv - Nb - derivatives of potential basis functions
       dv[j] : Nd -> Nd
       (optional)
       m_ - Nd x Nd - (constant) mass matrix / inertia tensor
       m : Nd -> Nd x Nd - (configuration-dependent) mass matrix / inertia tensor

    Outputs:
        c - Nb x Nd - coefficients for Nb potential basis functions in Nd variables
    """
    # reshape input
    q = np.asarray(q).T; a = np.asarray(a).T
    # q contains N observations of Nd dimensional configurations
    Nd,N = q.shape
    # a contains N observations of Nd dimensional accelerations
    assert a.shape == q.shape
    # monomials defining potential function
    #Nb = len(v)
    # sanity check inertia tensor
    if m_ is None and m is None:
        I = np.identity(Nd)
        M = lambda _ : I
        if m_ is not None:
            assert m_.shape == (Nd,Nd)
            M = lambda _ : m_
    # form composite force vector and derivative matrix
    F = []; D = []
    qj = q[:,0:1]; aj = a[:,0:1]
    F = np.array(np.dot(M(qj),aj))
    D = np.array([dvb(*qj) for dvb in dv]).T
    for j in range(1, N):
        qj = q[:,j:j+1]; aj = a[:,j:j+1]
        #F.append(sum(np.dot(M(qj),aj)))
        F = np.vstack((F, np.dot(M(qj),aj)))
        #D.append(np.asarray([dvb(*qj).flatten() for dvb in dv]).T)
        D = np.vstack((D, np.array([dvb(*qj) for dvb in dv]).T))
    #F = np.asarray(F); D = np.asarray(D)
    # solve D c = F for potential basis function coefficients c
    D = np.matrix(D)
    F = np.matrix(F)
    c = np.linalg.inv(D.T*D)*D.T*F
    return c

def potentialFunction(dimension, maxPower, minPower):
    monomialList = poly.monomials(dimension, minPower, n=maxPower)
    #monomialList = [(2,0,0), (0,2,0), (0,0,2)]
    variables = []

    for i in range(dimension):
        variables.append(sympy.Symbol(string.ascii_lowercase[i]))
        setattr(sympy, string.ascii_lowercase[i], variables[-1])

    V = []

    for monomialTuple in monomialList:
        v = '1'
        for i in range(dimension):
            v += ' * ' + str(variables[i]) + '**' + str(monomialTuple[i])
        V.append(v)

    for i, v in enumerate(V):
        V[i] = sympy.sympify(v)

    #print V.subs([(sympy.a, 5), (sympy.b, 2), (sympy.c,3)]).evalf()
    return V, variables

def evaluatePotentialFunction(V, variables, values):
    valuesList = []

    for i, value in enumerate(values):
        valuesList.append((getattr(sympy, variables[i]), value))

    return V.subs(valuesList).evalf()

def gradientFunctions(V, variables):
    gradientFunctionsList = []

    for v in V:
        _ = []
        for variable in variables:
            _.append(sympy.diff(v, variable))
        gradientFunctionsList.append(_)

    return gradientFunctionsList

def gradientFunctionsToLambdas(dV, variables):
    lList = []

    functionHeader = str(variables[0])
    for i in range(1, len(variables)):
        functionHeader += ', ' + str(variables[i])
    functionHeader = 'lambda ' + functionHeader + ': '

    for pV in dV:
        l =  eval(functionHeader + str(pV))
        lList.append(l)

    return lList

'''
def evaluateGradientFunction(dv, c, q):
  accs = np.asarray([cb * dvb(*q) for cb,dvb in zip(c,dv)])
  return tuple(np.sum(accs,axis=0))
'''

def evaluateGradientFunction(lList, c, q):
   accs = zip(*[[c[i]* x for x in lList[i](*q)] for i, ci in enumerate(c)])
   return tuple([sum(acc) for acc in accs])

def loadPWHamil(label):
    return pickle.load(open(os.path.join(os.getcwd(), 'PWHamil',label+'.p'), 'rb'))

def loadEstimatesTable(label):
    return pickle.load(open(os.path.join(os.getcwd(), 'Estimates',label+'.p'), 'rb'))

def loadStanceInfo(label):
    return pickle.load(open(os.path.join(os.getcwd(), 'StanceInfo',label+'.p'), 'rb'))

def stanceInfo(data, dataIDs, template, obsID, label):
    results = {}
    qList, accelList, stanceDict, stepDict, tarsusStepList = footlocation.createPWHamilInput(data, dataIDs, template, obsID, 'left')
    results['qListLeft'] = qList
    results['accelListLeft'] = accelList
    results['stanceDictLeft'] = stanceDict
    results['stepDictLeft'] = stepDict
    results['tarsusStepListLeft'] = tarsusStepList
    qList, accelList, stanceDict, stepDict, tarsusStepList = footlocation.createPWHamilInput(data, dataIDs, template, obsID, 'right')
    results['qListRight'] = qList
    results['accelListRight'] = accelList
    results['stanceDictRight'] = stanceDict
    results['stepDictRight'] = stepDict
    results['tarsusStepListRight'] = tarsusStepList
    pickle.dump(results, open(os.path.join(os.getcwd(), 'stanceinfo',label+'.p'), 'wb'))

def runPWHamil(data, dataIDs, template, stanceInfoLabel, label):
    results = {}
    stanceDict = loadStanceInfo(stanceInfoLabel)
    V, variables = potentialFunction(3, 1, 2)
    #print evaluatePotentialFunction(V, variables, [5, 2, 3])
    dV = gradientFunctions(V, variables)
    lList = gradientFunctionsToLambdas(dV, variables)
    results['dV'] = dV
    results['variables'] = variables
    cLeft = fit(stanceDict['qListLeft'], stanceDict['accelListLeft'], lList)
    results['cLeft'] = np.asarray(cLeft).reshape(-1)
    cRight = fit(stanceDict['qListRight'], stanceDict['accelListRight'], lList)
    results['cRight'] = np.asarray(cRight).reshape(-1)
    pickle.dump(results, open(os.path.join(os.getcwd(), 'PWHamil',label+'.p'), 'wb'))
    return results

def estimatesTableStanceInfo(estimatesLabel, label):
    estimatesTable = loadEstimatesTable(estimatesLabel)
    results = {}
    qListLeft = deque()
    accelListLeft = deque()
    qListRight = deque()
    accelListRight = deque()
    for i in range(0, 32):
        qListLeft.append((estimatesTable.ix[i, 'Roach_x_KalmanX_Mean'], estimatesTable.ix[i, 'Roach_y_KalmanX_Mean'], estimatesTable.ix[i, 'Roach_theta_KalmanX_Mean']))
        accelListLeft.append((estimatesTable.ix[i, 'Roach_x_KalmanDDX_Mean'], estimatesTable.ix[i, 'Roach_y_KalmanDDX_Mean'], estimatesTable.ix[i, 'Roach_theta_KalmanDDX_Mean']))
    results['qListLeft'] = qListLeft
    results['accelListLeft'] = accelListLeft
    for i in range(32, 64):
        qListRight.append((estimatesTable.ix[i, 'Roach_x_KalmanX_Mean'], estimatesTable.ix[i, 'Roach_y_KalmanX_Mean'], estimatesTable.ix[i, 'Roach_theta_KalmanX_Mean']))
        accelListRight.append((estimatesTable.ix[i, 'Roach_x_KalmanDDX_Mean'], estimatesTable.ix[i, 'Roach_y_KalmanDDX_Mean'], estimatesTable.ix[i, 'Roach_theta_KalmanDDX_Mean']))
    results['qListRight'] = qListRight
    results['accelListRight'] = accelListRight
    pickle.dump(results, open(os.path.join(os.getcwd(), 'StanceInfo',label+'.p'), 'wb'))
    return results

def runPWHamilEstimates(data, dataIDs, template, estimatesLabel, label):
    results = {}
    V, variables = potentialFunction(3, 1, 2)
    stanceDict = estimatesTabletoPWHamilInput(estimatesLabel)
    #print evaluatePotentialFunction(V, variables, [5, 2, 3])
    dV = gradientFunctions(V, variables)
    lList = gradientFunctionsToLambdas(dV, variables)
    results['dV'] = dV
    results['variables'] = variables
    cLeft = fit(stanceDict['qListLeft'], stanceDict['accelListLeft'], lList)
    results['cLeft'] = np.asarray(cLeft).reshape(-1)
    cRight = fit(stanceDict['qListRight'], stanceDict['accelListRight'], lList)
    results['cRight'] = np.asarray(cRight).reshape(-1)
    pickle.dump(results, open(os.path.join(os.getcwd(), 'PWHamil',label+'.p'), 'wb'))
    return results

'''
def transformation1(feetX, feetY, bodyX, bodyY, yaw):
    fx = []
    fy = []

    for feet in feetX:
        fx.append(feet)
    for feet in feetY:
        fy.append(feet)

    zBody = bodyX + 1.j * bodyY
    zFoot = pd.Series(fx).mean() + 1.j * pd.Series(fy).mean()
    zFootTransform = zBody + zFoot * np.exp(1.j * yaw)
    #TODO: USE CORRECT FRAME
    fxy = np.matrix([fx, fy])
    U, s, V = np.linalg.svd(fxy)
    thetaFootAvg = np.arctan2(U.item(1,0), U.item(0,0))
    if abs(thetaFootAvg) > np.pi/2:
        thetaFootAvg = np.arctan2(-1.0 * U.item(1,0), -1.0 * U.item(0,0))

    zBodyTransform = zBody * np.exp(1.j * (yaw - thetaFootAvg)) - zFoot
    yawTransform = yaw - thetaFootAvg
    return zBodyTransform.real, zBodyTransform.imag, yawTransform, zFootTransform.real, zFootTransform.imag, thetaFootAvg
'''

def transformation1(x_fb, y_fb, x_bc, y_bc, theta_bc):
    U, s, V = np.linalg.svd(np.matrix([x_fb, y_fb]))
    theta_fb = np.arctan2(U.item(1,0), U.item(0,0))
    if abs(theta_fb) > np.pi/2:
        theta_fb = np.arctan2(-1.0 * U.item(1,0), -1.0 * U.item(0,0))
    theta_fc = theta_fb + theta_bc
    theta_bf = -theta_fb
    z_bc = x_bc + 1.j * y_bc
    z_fb = pd.Series(x_fb).mean() + 1.j * pd.Series(y_fb).mean()
    z_fc = z_bc + z_fb * np.exp(1.j * theta_bc)
    z_bf = z_bc - z_fc
    return z_bf.real, z_bf.imag, theta_bf, z_fc.real, z_fc.imag, theta_fc

def transformation2(ddx_f, ddy_f, theta_fc):
    ddz_f = ddx_f + 1.j * ddy_f
    ddz_fc = ddz_f * np.exp(1.j * theta_fc)
    return ddz_fc.real, ddz_fc.imag

def generatePWHamilTable(kalmanLabel, pwHamilLabel):
    pwHamilResults = loadPWHamil(pwHamilLabel)
    estimatesTable = loadEstimatesTable(kalmanLabel)
    lList = gradientFunctionsToLambdas(pwHamilResults['dV'], pwHamilResults['variables'])

    pwHamilTable = pd.DataFrame(index=range(64), columns=['DDX', 'DDY', 'DDTheta'])

    (x, _, accelX) = kalman.columnTableX('Roach_x')
    (y, _, accelY) = kalman.columnTableX('Roach_y')
    (theta, _, accelTheta) = kalman.columnTableX('Roach_theta')

    (fl1X, _, _) = kalman.columnTableX('TarsusBody1_x')
    (fl2X, _, _) = kalman.columnTableX('TarsusBody2_x')
    (fl3X, _, _) = kalman.columnTableX('TarsusBody3_x')
    (fl4X, _, _) = kalman.columnTableX('TarsusBody4_x')
    (fl5X, _, _) = kalman.columnTableX('TarsusBody5_x')
    (fl6X, _, _) = kalman.columnTableX('TarsusBody6_x')

    (fl1Y, _, _) = kalman.columnTableX('TarsusBody1_y')
    (fl2Y, _, _) = kalman.columnTableX('TarsusBody2_y')
    (fl3Y, _, _) = kalman.columnTableX('TarsusBody3_y')
    (fl4Y, _, _) = kalman.columnTableX('TarsusBody4_y')
    (fl5Y, _, _) = kalman.columnTableX('TarsusBody5_y')
    (fl6Y, _, _) = kalman.columnTableX('TarsusBody6_y')

    x = estimatesTable[x + '_Mean']
    y = estimatesTable[y + '_Mean']
    theta = estimatesTable[theta + '_Mean']

    accelX = estimatesTable[accelX + '_Mean']
    accelY = estimatesTable[accelY + '_Mean']
    accelTheta = estimatesTable[accelTheta + '_Mean']

    fl1X = estimatesTable[fl1X + '_Mean']
    fl2X = estimatesTable[fl2X + '_Mean']
    fl3X = estimatesTable[fl3X + '_Mean']
    fl4X = estimatesTable[fl4X + '_Mean']
    fl5X = estimatesTable[fl5X + '_Mean']
    fl6X = estimatesTable[fl6X + '_Mean']

    fl1Y = estimatesTable[fl1Y + '_Mean']
    fl2Y = estimatesTable[fl2Y + '_Mean']
    fl3Y = estimatesTable[fl3Y + '_Mean']
    fl4Y = estimatesTable[fl4Y + '_Mean']
    fl5Y = estimatesTable[fl5Y + '_Mean']
    fl6Y = estimatesTable[fl6Y + '_Mean']

    for i in range(0, 32):
        bodyX, bodyY, bodyTheta, xFoot, yFoot, thetaFoot = transformation1([fl1X.ix[i], fl3X.ix[i], fl5X.ix[i]],  \
        [fl1Y.ix[i], fl3Y.ix[i], fl5Y.ix[i]], x.ix[i], y.ix[i], theta.ix[i])
        (xAccelPW, yAccelPW, thetaAccelPW) = evaluateGradientFunction(lList, pwHamilResults['cLeft'], (bodyX, bodyY, bodyTheta))
        ddx_fc, ddy_fc = transformation2(xAccelPW, yAccelPW, thetaFoot)
        pwHamilTable.ix[i, 'DDX'] = ddx_fc
        pwHamilTable.ix[i, 'DDY'] = ddy_fc
        pwHamilTable.ix[i, 'DDTheta'] = thetaAccelPW

    for i in range(32, 64):
        bodyX, bodyY, bodyTheta, xFoot, yFoot, thetaFoot = transformation1([fl2X.ix[i], fl4X.ix[i], fl6X.ix[i]],  \
        [fl2Y.ix[i], fl4Y.ix[i], fl6Y.ix[i]], x.ix[i], y.ix[i], theta.ix[i])
        (xAccelPW, yAccelPW, thetaAccelPW) = evaluateGradientFunction(lList, pwHamilResults['cRight'], (bodyX, bodyY, bodyTheta))
        ddx_fc, ddy_fc = transformation2(xAccelPW, yAccelPW, thetaFoot)
        pwHamilTable.ix[i, 'DDX'] = ddx_fc
        pwHamilTable.ix[i, 'DDY'] = ddy_fc
        pwHamilTable.ix[i, 'DDTheta'] = thetaAccelPW

    return pwHamilTable

def plotPWHamil(label, color, pwHamilTable, kalmanTable):
    (_, _, accelX) = kalman.columnTableX('Roach_x')
    plt.figure(1,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(pwHamilTable.index, pwHamilTable['DDX'], 'k' + '.-', lw=3., markersize=12, label='pwHamil Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelX + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelX + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelX + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddx (cm/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('PWHamil/' + label + '-ddx.svg')

    (_, _, accelY) = kalman.columnTableX('Roach_y')
    plt.figure(2,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(pwHamilTable.index, pwHamilTable['DDY'], 'k' + '.-', lw=3., markersize=12, label='pwHamil Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelY + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelY + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelY + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddy (cm/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('PWHamil/' + label + '-ddy.svg')

    (_, _, accelTheta) = kalman.columnTableX('Roach_theta')
    plt.figure(3,figsize=(12,8)); plt.clf()
    dataLegend = plt.plot(pwHamilTable.index, pwHamilTable['DDTheta'], 'k' + '.-', lw=3., markersize=12, label='pwHamil Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelTheta + '_Mean'], color + '.-', lw=3., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelTheta + '_01'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    dataLegend = plt.plot(kalmanTable.index, kalmanTable[accelTheta + '_99'], color + '-', lw=1., markersize=12, label='kalman Estimate')
    #plt.legend(handles=[dataLegend], loc=2, prop={'size':12})
    plt.xlabel('phase (radians)')
    plt.xlim(0,64)
    plt.ylabel('ddtheta (deg/s**2)')
    plt.grid(True)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    plt.tight_layout()
    plt.savefig('PWHamil/' + label + '-ddtheta.svg')

def runPlotPWHamil():
    pwHamilTableControl = generatePWHamilTable('control', 'control')
    pwHamilTableMass = generatePWHamilTable('mass', 'mass')
    pwHamilTableInertia = generatePWHamilTable('inertia', 'inertia')
    estimatesControl = loadEstimatesTable('control')
    estimatesMass = loadEstimatesTable('mass')
    estimatesInertia = loadEstimatesTable('inertia')
    plotPWHamil('control-150903', 'b', pwHamilTableControl, estimatesControl)
    plotPWHamil('mass-150903', 'r', pwHamilTableMass, estimatesMass)
    plotPWHamil('inertia-150903', 'g', pwHamilTableInertia, estimatesInertia)

if __name__ == "__main__":
    saveDir = 'StableOrbit'

    mw = model.ModelWrapper(saveDir)
    mo = model.ModelOptimize(mw)
    mc = model.ModelConfiguration(mw)


    results = estimatesTableStanceInfo('control', 'control-estimates')
    results = estimatesTableStanceInfo('mass', 'mass-estimates')
    results = estimatesTableStanceInfo('inertia', 'inertia-estimates')


    #Generate stance info
    '''
    mw.csvLoadObs([0, 1, 2])
    template = mc.jsonLoadTemplate('templateControl')
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    stanceInfo(mw.data, dataIDs, template, 0, 'control')
    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    mw.csvLoadData(dataIDs)
    stanceInfo(mw.data, dataIDs, template, 1, 'mass')
    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    mw.csvLoadData(dataIDs)
    stanceInfo(mw.data, dataIDs, template, 2, 'inertia')
    '''

    '''
    #dataIDs = mo.treatments.query("Treatment == 'control' and AnimalID == 2").index
    dataIDs = mo.treatments.query("Treatment == 'control'").index
    mw.csvLoadData(dataIDs)
    template = mc.jsonLoadTemplate('templateControl')
    runPWHamil(mw.data, dataIDs, template, 'control', 'control')

    dataIDs = mo.treatments.query("Treatment == 'mass'").index
    mw.csvLoadData(dataIDs)
    template = mc.jsonLoadTemplate('templateMass')
    runPWHamil(mw.data, dataIDs, template, 'mass', 'mass')

    dataIDs = mo.treatments.query("Treatment == 'inertia'").index
    mw.csvLoadData(dataIDs)
    template = mc.jsonLoadTemplate('templateInertia')
    runPWHamil(mw.data, dataIDs, template, 'inertia', 'inertia')

    runPlotPWHamil()
    '''

    pwHamilTableControl = generatePWHamilTable('control-raw', 'control')
    pwHamilTableMass = generatePWHamilTable('mass-raw', 'mass')
    pwHamilTableInertia = generatePWHamilTable('inertia-raw', 'inertia')

    '''
    V, variables = potentialFunction(3, 1, 2)
    #print evaluatePotentialFunction(V, variables, [5, 2, 3])
    dV = gradientFunctions(V, variables)
    lList = gradientFunctionsToLambdas(dV, variables)
    qList, accelList = footlocation.createPWHamilInput(mw.data, dataIDs, template, 'left')
    cLeft = fit(qList, accelList, lList)
    qList, accelList = footlocation.createPWHamilInput(mw.data, dataIDs, template, 'right')
    cRight = fit(qList, accelList, lList)

    c = np.array(cLeft)
    c = c.flatten()
    K = np.asarray([[.5*c[8],c[7],c[6]],[0.,.5*c[5],c[4]],[0.,0.,c[3]]])
    K = K + K.T
    Kq = np.asarray([[c[2],c[1],c[0]]]).T
    KLeft = K
    qLeft = np.dot(np.linalg.inv(K),Kq)

    c = np.array(cRight)
    c = c.flatten()
    K = np.asarray([[.5*c[8],c[7],c[6]],[0.,.5*c[5],c[4]],[0.,0.,c[3]]])
    K = K + K.T
    Kq = np.asarray([[c[2],c[1],c[0]]]).T
    KRight = K
    qRight = np.dot(np.linalg.inv(K),Kq)

    print KLeft
    print
    print KRight
    print
    print qLeft
    print
    print qRight
    '''

    #Sam's Stuff
    '''
    sum([c[j] * lList[j](*q) for j in range(len(c))])



    Nd = 3
    deg = 2

    val = lambda p,q : poly.val(p[0],p[1],q)

    v = poly.monomials(Nd,deg,n=1)

    #v = [v[3]]
    Nb = len(v)
    dv_ = [[poly.diff([vb],[[1.]],j) for j in range(Nd)] for vb in v]
    dv = [lambda q : np.asarray([val(poly.diff([vb],[[1.]],j),q).flatten()
                    for j in range(Nd)])
        for vb in v]

    #c = np.asarray([0.,0.,1.,1.,1.])
    #c = np.asarray([1.,1.,0.,0.,0.])
    #c1 = np.asarray([1.,0.])
    #c2 = np.asarray([0.,1.])
    #c = np.asarray([0.,0.,0.,1.,1.])
    c = np.asarray([1.,1.])

    q = np.linspace(-1.,1.,5)[np.newaxis,...]
    q0 = q[:,0:1]
    q0 = np.asarray([[-1.,0.]]).T
    if 0:
        vqc1 = poly.val(v,c1,q)
        vqc2 = poly.val(v,c2,q)
        #dv_ = [val(poly.diff(

    plt.figure(1); plt.clf()

    ax = plt.subplot(2,1,1)
    ax.plot(q[0],vqc1)
    ax.plot(q[0],vqc2)
    ax.set_ylabel('$v$')

    ax = plt.subplot(2,1,2)
    ax.set_ylabel('$v$')
    '''
