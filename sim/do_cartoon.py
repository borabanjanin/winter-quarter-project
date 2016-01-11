import numpy as np
import pylab as plt
import matplotlib.patches as patches
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

D = pickle.load(open( "Estimates/control.p", "rb" ))
print D
#D['control']

sp = 35. # cm / sec
f = 11. # strides / sec
fps = 500 # frames / sec
l = 4.4; w = 1.75

N = 64
ph = np.linspace(0,2*np.pi,N)
ti = (ph/(2*np.pi)) / f


s = 1.
sdxy = np.sqrt((np.asarray(list(D[K['dx']])).real+sp)**2
             + (np.asarray(list(D[K['dy']])).real)**2).max()
sddxy = np.sqrt((np.asarray(list(D[K['ddx']])).real)**2
              + (np.asarray(list(D[K['ddy']])).real)**2).max()
sdt = np.abs(np.asarray(list(D[K['dt']])).real).max()
sddt = np.abs(np.asarray(list(D[K['ddt']])).real).max()

sdxy /= 2.; sddxy /= 2;

Hb = np.asarray([1.4+1.j*.35, .7 +1.j*.35, 0. +1.j*.35,
                 0. -1.j*.35, .7 -1.j*.35, 1.4-1.j*.35,])

L = []

global xlim,ylim
xlim = (-3.,6.); ylim = (-2.,2.)

def cartoon(D,K,n=None,ax=None,clf=False,lims=True,trans=True,rng=[-N/2,N/2]):
  global xlim,ylim
  body = dict(lw=6.,edgecolor='k',facecolor=.5*np.ones(3),alpha=0.5)
  hips = dict(ms=12.,color='k')
  feet0 = dict(ms=12.,color='r',mec='r',mew=2.)
  feet1 = dict(ms=12.,color='b',mec='b',mew=2.)
  vel = dict(lw=6.,color='#255c69',width=.25,alpha=1.,zorder=10)
  acc = dict(lw=6.,color='#aa3e39',width=.25,alpha=1.,zorder=10)
  if ax is None or clf:
    plt.figure(1,figsize=(8,4)); plt.clf()
    ax = plt.axes(); ax.axis('equal'); ax.grid('on');
    ax.autoscale('false')

  if n is None or clf:
    cartoon(D,K,rng[0],ax,lims=False)
    cartoon(D,K,rng[1],ax,lims=False)
    ax.relim()
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    for l in L:
      l.set_visible(False)
    #plt.draw()
    if n is None:
      return ax

  for k in K.keys():
    cmd = k+'=D["'+K[k]+'"]['+str(n%N)+'].real'
    exec cmd

  dx += sp
  #x += sp*ti[n]
  if trans:
    x += sp*(float(n)/N)/f
  #1/0
  x *= s; y *= s; t *= s
  zxy = x+1.j*y
  zt = np.exp(1.j*t)
  H = Hb*zt + zxy
  Zb = np.asarray([X1+1.j*Y1, X2+1.j*Y2, X3+1.j*Y3,
                   X4+1.j*Y4, X5+1.j*Y5, X6+1.j*Y6])
  Z = Zb*zt + zxy
  Z0 = Z[0::2]; Z1 = Z[1::2]

  dzxy = zt*(dx+1.j*dy)/sdxy
  ddzxy = zt*(ddx+1.j*ddy)/sddxy
  dzxyx = zt*(dx)/sdxy
  ddzxyx = zt*(ddx)/sddxy
  dzxyy = zt*(1.j*dy)/sdxy
  ddzxyy = zt*(1.j*ddy)/sddxy

  # body
  L.append(ax.add_patch(patches.Ellipse((x,y),4.4,1.75,t,**body)))
  # hips
  L.extend(ax.plot(H.real,H.imag,'o',**hips))
  # feet
  if n%N > N/2:
    feet0['mfc'] = 'w'
    L.extend(ax.plot(Z0.real,Z0.imag,'^',**feet0))
    feet1['mfc'] = 'b'
    L.extend(ax.plot(Z1.real,Z1.imag,'^',**feet1))
  else:
    feet0['mfc'] = 'r'
    L.extend(ax.plot(Z0.real,Z0.imag,'^',**feet0))
    feet1['mfc'] = 'w'
    L.extend(ax.plot(Z1.real,Z1.imag,'^',**feet1))
  # TODO: use FancyArrowPatch instead
  # vel
  L.append(ax.add_patch(patches.Arrow(x,y,dzxy.real,dzxy.imag,**vel)))
  #L.append(ax.add_patch(patches.Arrow(x,y,dzxyx.real,dzxyx.imag,**vel)))
  #L.append(ax.add_patch(patches.Arrow(x,y,dzxyy.real,dzxyy.imag,**vel)))
  # acc
  #L.append(ax.add_patch(patches.Arrow(x,y,ddzxy.real,ddzxy.imag,**acc)))
  L.append(ax.add_patch(patches.Arrow(x,y,ddzxyx.real,ddzxyx.imag,**acc)))
  L.append(ax.add_patch(patches.Arrow(x,y,ddzxyy.real,ddzxyy.imag,**acc)))

  if lims:
    ax.set_xlim(xlim); ax.set_ylim(ylim)
  return ax


rng = [-N,N]
ax = cartoon(D,K,rng=rng)

for n in range(*rng):
#for n in [-N/2,-N/4,0,N/4,N/2]:
#for n in [-10]:
#for n in rng:
  cartoon(D,K,n,ax,clf=True,rng=rng)
  #cartoon(D,K,n,ax)
  plt.draw()
  plt.show()


#ax.plot(p,d['control']['Roach_x_KalmanDX_Mean'],'b',lw=2)
#ax.plot(p,d['control']['Roach_x_KalmanDX_01'],'b--')
#ax.plot(p,d['control']['Roach_x_KalmanDX_99'],'b--')

#ax.set_xlabel('phase')
#ax.set_ylabel('fore-aft pos')
