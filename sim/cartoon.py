import numpy as np
import pylab as plt
import pickle

K = dict(
x='Roach_x_KalmanX_Mean',
y='Roach_y_KalmanX_Mean',
t='Roach_theta_KalmanX_Mean',
p='Roach_pitch_KalmanX_Mean',
r='Roach_roll_KalmanX_Mean',
X1='TarsusBody1_x_KalmanX_Mean',
X2='TarsusBody2_x_KalmanX_Mean',
X3='TarsusBody3_x_KalmanX_Mean',
X4='TarsusBody4_x_KalmanX_Mean',
X5='TarsusBody5_x_KalmanX_Mean',
X6='TarsusBody6_x_KalmanX_Mean',
Y1='TarsusBody1_y_KalmanX_Mean',
Y2='TarsusBody2_y_KalmanX_Mean',
Y3='TarsusBody3_y_KalmanX_Mean',
Y4='TarsusBody4_y_KalmanX_Mean',
Y5='TarsusBody5_y_KalmanX_Mean',
Y6='TarsusBody6_y_KalmanX_Mean'
)

for k in K.keys():
  K['d'+k]  = K[k].replace('X','DX')
  K['dd'+k] = K[k].replace('X','DDX')

for k in K.keys():
  K[k+'m'] = K[k].replace('Mean','01')
  K[k+'M'] = K[k].replace('Mean','99')

d = pickle.load(open( "estimates.p", "rb" ))
d['control']

n = 64
p = np.linspace(0,2*np.pi,n)

plt.figure(1); plt.clf()
ax = plt.subplot(1,1,1)

#ax.plot(p,d['control']['Roach_x_KalmanDX_Mean'],'b',lw=2)
#ax.plot(p,d['control']['Roach_x_KalmanDX_01'],'b--')
#ax.plot(p,d['control']['Roach_x_KalmanDX_99'],'b--')

#ax.set_xlabel('phase')
#ax.set_ylabel('fore-aft pos')
