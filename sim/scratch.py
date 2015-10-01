
if 0:
  # transform 2d data between reference frames
  # xb,yb - tarsus position
  # x,y,th - body pos, angle
  zb = xb + 1.j*yb
  z = x + 1.j*y
  zc = z + zb * np.exp(1.j*th)
  xc,yc = zc.real,zc.imag
  assert th == np.angle(np.exp(1.j*th)), "np.angle yields angle of complex number"


import numpy as np
import pylab as plt
from util import poly
from util.poly import monomials, fit, val, diff, jac


# scalar problem data
deg = 2; Nq = 1; Nv = 1; N = 100; s = 1e-0
# (x,y,theta) data
#deg = 2; Nq = 3; Nv = 1; N = 100; s = 1e-0
# (x,y,yaw,pitch,roll) data
#deg = 2; Nq = 5; Nv = 1; N = 100; s = 1e-0

# monomials whose coefficients we seek to fit
p = monomials(Nq,deg,n=2)

# coefficients we seek to fit
a = np.random.randn(Nv,len(p))*np.round(np.random.rand(Nv,len(p)))

# N samples in the dataset
q = np.random.randn(Nq,N)

# noisy values of polynomial we seek to fit
v = val(p,a,q) + s*np.random.randn(Nv,N)
# outliers should cause big problems!
#q[0,0] = -2 
#v[0,0] = 5

# fit polynomial using noisy observations
_,aa = fit(q,v,p=p)

print ('%d order polv, %d vars, %d outputs, %d training samples' 
        % (deg,Nq,Nv,N))
print '(aa)',   (aa) 
print '(a)',    (a) 
print '(aa-a)', (aa-a) 

_ = np.linspace(-3.,3.)

plt.figure(1); plt.clf()
plt.plot(_,val(p,a,np.asarray([_])).flatten(),'g',lw=4)
plt.plot(q,v,'k.')
plt.plot(_,val(p,aa,np.asarray([_])).flatten(),'r--',lw=2)
