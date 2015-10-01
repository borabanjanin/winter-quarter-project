import modelwrapper as model
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import copy
import pickle


class Spring(object):
    #Spring undefined when the x coordinates are equal
    def __init__(self, k, length, x0, y0, x1, y1):
        self.k = k
        self.length = length
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def __call__(self):
        currentLength = np.sqrt((self.y1 - self.y0)**2 + (self.x1 - self.x0)**2)
        forceMagnitude = .5 * self.k * (currentLength - self.length)**2
        forceAngle = np.arctan2(np.abs(self.y1 - self.y0), np.abs(self.x1 - self.x0))
        #undefined behavior
        if self.y1 == self.y0 and self.x1 == self.x0:
            #raise Exception('Spring undefined when contracted to 0 length')
            return (np.nan, np.nan)
        return (np.cos(forceAngle) * forceMagnitude, np.sin(forceAngle) * forceMagnitude)

class TriLLS(object):
    def __init__(self, k, length, x0, y0, x1, y1):
        self.k = k
        self.length = length
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 =y1
        #initialize springs
        self.setSprings()

    def setSprings(self):
        self.spring1 = Spring(self.k[0], self.length[0], self.x0[0], self.y0[0], self.x1[0], self.y1[0])
        self.spring2 = Spring(self.k[1], self.length[1], self.x0[1], self.y0[1], self.x1[1], self.y1[1])
        self.spring3 = Spring(self.k[2], self.length[2], self.x0[2], self.y0[2], self.x1[2], self.y1[2])

    def __call__(self):
        (fx1, fy1, tz1) = self.translateSpringForce(self.spring1)
        (fx2, fy2, tz2) = self.translateSpringForce(self.spring2)
        (fx3, fy3, tz3) = self.translateSpringForce(self.spring3)
        return (fx1 + fx2 + fx3, fy1 + fy2 + fy3, tz1 + tz2 + tz3)

    def translateSpringForce(self, spring):
        zeroMatrix = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        pbc = np.matrix([[spring.x0, spring.y0, 0]])
        phat = np.matrix([[0, -pbc.item(0,2), pbc.item(0,1)], [pbc.item(0,2), 0, -pbc.item(0,0)], [-pbc.item(0,1), pbc.item(0,0), 0]])
        ft = np.matrix([[spring()[0]], [spring()[1]], [0], [0], [0], [0]])
        transformation = np.vstack([np.hstack([model.ModelWrapper.rotMat3(0).transpose(), \
        zeroMatrix]),np.hstack([-model.ModelWrapper.rotMat3(0).transpose()*phat, model.ModelWrapper.rotMat3(0).transpose()])])
        fc = transformation * ft
        return (fc.item(0,0), fc.item(1,0), fc.item(5,0))

def plotContourTrills(trills, var1, var2, xRange, yRange, thetaRange):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    RESOLUTION = 14
    x = np.arange(xRange[0], xRange[1], (xRange[1] - xRange[0])/RESOLUTION)
    y = np.arange(yRange[0], yRange[1], (yRange[1] - yRange[0])/RESOLUTION)
    theta = np.arange(thetaRange[0], thetaRange[1], (thetaRange[1] - thetaRange[0])/RESOLUTION)
    X1, Y1 = np.meshgrid(x, y)
    X2, Theta2 = np.meshgrid(x, theta)
    Y3, Theta3 = np.meshgrid(y, theta)
    (height, width) = np.shape(X1)
    XY = np.zeros((height, width))
    XTheta = np.zeros((height, width))
    YTheta = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            trillsTemp = copy.deepcopy(trills)
            setattr(trillsTemp, var1, [x + X1[i,j] for x in getattr(trills, var1)])
            setattr(trillsTemp, var2, [x + Y1[i,j] for x in getattr(trills, var2)])
            trillsTemp.setSprings()
            XY[i,j] = trillsTemp()[0] + trillsTemp()[1]

            trillsTemp = copy.deepcopy(trills)
            setattr(trillsTemp, var1, [X2[i,j] for x in getattr(trills, var1)])
            setattr(trillsTemp, var1, [x*np.cos(Theta2[i,j]) - y*np.sin(Theta2[i,j]) for x, y in zip(getattr(trills, var1), getattr(trills, var2))])
            setattr(trillsTemp, var1, [x*np.sin(Theta2[i,j]) + y*np.cos(Theta2[i,j]) for x, y in zip(getattr(trills, var1), getattr(trills, var2))])
            trillsTemp.setSprings()
            XTheta[i,j] = trillsTemp()[0] + trillsTemp()[1]

            trillsTemp = copy.deepcopy(trills)
            setattr(trillsTemp, var1, [Y3[i,j] for x in getattr(trills, var2)])
            setattr(trillsTemp, var1, [x*np.cos(Theta3[i,j]) - y*np.sin(Theta3[i,j]) for x, y in zip(getattr(trills, var1), getattr(trills, var2))])
            setattr(trillsTemp, var1, [x*np.sin(Theta3[i,j]) + y*np.cos(Theta3[i,j]) for x, y in zip(getattr(trills, var1), getattr(trills, var2))])
            trillsTemp.setSprings()
            YTheta[i,j] = trillsTemp()[0] + trillsTemp()[1]

    plt.figure(1)
    CS = plt.contour(X1, Y1, XY)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('X vs Y')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(2)
    CS = plt.contour(X2, Theta2, XTheta)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('X vs Theta')
    plt.xlabel('x')
    plt.ylabel('theta')

    plt.figure(3)
    CS = plt.contour(Y3, Theta3, YTheta)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Y vs Theta')
    plt.xlabel('y')
    plt.ylabel('theta')
    plt.show()

    return XY, XTheta, YTheta, trills

def plotTrillsData(label, roach):
    tarsusData = pickle.load(open( "tarsusData.p", "rb" ))
    currentData = tarsusData[label]
    t1x = currentData['1x']['TarsusBody1_x_KalmanX_Mean']
    t3x = currentData['3x']['TarsusBody3_x_KalmanX_Mean']
    t5x = currentData['5x']['TarsusBody5_x_KalmanX_Mean']
    t1y = currentData['1y']['TarsusBody1_y_KalmanX_Mean']
    t3y = currentData['3y']['TarsusBody3_y_KalmanX_Mean']
    t5y = currentData['5y']['TarsusBody5_y_KalmanX_Mean']

    t2x = currentData['2x']['TarsusBody2_x_KalmanX_Mean']
    t4x = currentData['4x']['TarsusBody4_x_KalmanX_Mean']
    t6x = currentData['6x']['TarsusBody6_x_KalmanX_Mean']
    t2y = currentData['2y']['TarsusBody2_y_KalmanX_Mean']
    t4y = currentData['4y']['TarsusBody4_y_KalmanX_Mean']
    t6y = currentData['6y']['TarsusBody6_y_KalmanX_Mean']

    plt.figure(1);plt.clf();
    plt.plot(currentData['1x']['TarsusBody1_x_KalmanX_Mean'], '.')
    plt.plot(currentData['3x']['TarsusBody3_x_KalmanX_Mean'], '.')
    plt.plot(currentData['5x']['TarsusBody5_x_KalmanX_Mean'], '.')

    plt.plot(range(32, 64) + range(0,32), currentData['2x']['TarsusBody2_x_KalmanX_Mean'], '.')
    plt.plot(range(32, 64) + range(0,32), currentData['4x']['TarsusBody4_x_KalmanX_Mean'], '.')
    plt.plot(range(32, 64) + range(0,32), currentData['6x']['TarsusBody6_x_KalmanX_Mean'], '.')
    plt.show()

    plt.figure(2);plt.clf();
    plt.plot(-currentData['1y']['TarsusBody1_y_KalmanX_Mean'], '.')
    plt.plot(-currentData['3y']['TarsusBody3_y_KalmanX_Mean'], '.')
    plt.plot(currentData['5y']['TarsusBody5_y_KalmanX_Mean'], '.')

    plt.plot(range(32, 64) + range(0,32), -currentData['2y']['TarsusBody2_y_KalmanX_Mean'], '.')
    plt.plot(range(32, 64) + range(0,32), currentData['4y']['TarsusBody4_y_KalmanX_Mean'], '.')
    plt.plot(range(32, 64) + range(0,32), currentData['6y']['TarsusBody6_y_KalmanX_Mean'], '.')
    plt.show()

    potentialEnergy = []

    for i in range(0, 32):
        roach.x1 = (t1x.ix[i].real, t3x.ix[i].real, t5x.ix[i].real)
        roach.y1 = (t1y.ix[i].real, t3y.ix[i].real, t5y.ix[i].real)
        roach.setSprings()
        (xEnergy, yEnergy, zEnergy) = roach()
        potentialEnergy.append(np.sqrt(xEnergy**2 + yEnergy**2))

    roach.x0 = (1.4, 0.7, 0.0)
    roach.y0 = (-0.35, 0.35, -0.35)

    for i in range(32, 64):
        roach.x1 = (t6x.ix[i].real, t4x.ix[i].real, t2x.ix[i].real)
        roach.y1 = (t6y.ix[i].real, t4y.ix[i].real, t2y.ix[i].real)
        roach.setSprings()
        (xEnergy, yEnergy, zEnergy) = roach()
        potentialEnergy.append(np.sqrt(xEnergy**2 + yEnergy**2))

    plt.figure(3); plt.clf();
    plt.plot(potentialEnergy)
    plt.show()

    return currentData

if __name__ == '__main__':

    #physically realistic values
    x0 = (1.4, 0.7, 0.0)
    y0 = (0.35, -0.35, 0.35)
    x1 = (-1.1, 1.3, -1.3)
    y1 = (2.0, 0.7, -1.0)

    k = (1, 1, 1)
    length = (1.218236430254817, 1.7204650534085254, 2.2360679774997897)
    roach = TriLLS(k, length, x0, y0, x1, y1)

    #XY, XTheta, YTheta , roach = plotContourTrills(roach, 'x1', 'y1', (-7.5, 7.5), (-7.5, 7.5), (-np.pi, np.pi))

    a = plotTrillsData('control', roach)

    #testing spring force translation
    '''
    x0 = (5, 0, 0)
    y0 = (1, 0, 0)
    x1 = (5, 0, 0)
    y1 = (0, 0, 0)
    k = (.25, 0, 0)
    length = (1, 0, 0)

    values = []
    testPoints = [-25, -10, -5, 5, 10, 25]
    torques = []

    for i in testPoints:
        y0 = (i, 0, 0)
        roach = TriLLS(k, length, x0, y0, x1, y1)
        values.append(roach())


    plt.figure(1); plt.clf()
    for i, testPoint in enumerate(testPoints):
        currentValue = values[i]
        plt.plot([x0[0], currentValue[0] + x0[0]], [testPoint, currentValue[1] + testPoint], '.-', lw=3)
        torques.append(currentValue[2])
    plt.show()

    plt.figure(2); plt.clf();
    plt.plot(testPoints, torques, '.-', lw=3)
    plt.show()
    '''
