
import numpy as np
import pylab as plt

from shrevz import util as shutil

def f(ph):
  return np.cos(ph + 3*np.pi/5)

n = 1000
sph = 1e-1
sfph = 1e-0
ph = np.arange(0,1000)*10/(2*np.pi) + np.cumsum(sph*np.random.randn(n))
fph = f(ph) + sfph*np.random.randn(n)

ord = 3
fs = shutil.FourierSeries()
fs.fit(ord,ph[np.newaxis,:],fph[np.newaxis,:])

ph0 = np.arange(0,2*np.pi,2*np.pi/(2**14))
fph0 = fs.val(ph0)

fig = plt.figure(1); fig.clf();
ax = plt.subplot(1,1,1); ax.grid('on');

plt.plot(np.mod(ph,2*np.pi),fph,'.')
plt.plot(np.mod(ph0,2*np.pi),f(ph0),'r.-',lw=4.)
plt.plot(np.mod(ph0,2*np.pi),fph0,'g.-',lw=2.)

