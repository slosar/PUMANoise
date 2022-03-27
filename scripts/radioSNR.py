#!/usr/bin/env python
from pumanoise import PUMA, PUMAPetite
import matplotlib.pyplot as plt
import pyccl as ccl
import numpy as np
from matplotlib.colors import LogNorm


C=ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)
puma=PUMA(C)
pumapt=PUMAPetite(C)

hub=C['h']
k=np.array([0.2*hub,0.4*hub,0.8*hub])
mu=0.5
kpar=np.outer(k*mu,np.ones(3))
kperp=np.outer(np.ones(3),k*np.sqrt(1-mu**2))

zs=np.linspace(0.3,6,100)
res,respt=[],[]
for z in zs:
    f=ccl.growth_rate(C,1/(1+z))
    PkReal=ccl.linear_matter_power(C,k,1/(1+z))
    Pk=puma.Tb(z)**2 * PkReal * (puma.bias(z) + f* mu**2)**2
    noise=puma.PNoiseKFull(z,kperp,kpar)
    noisept=pumapt.PNoiseKFull(z,kperp,kpar)
    res.append((Pk/noise).diagonal())
    respt.append((Pk/noisept).diagonal())

res=np.array(res)
respt=np.array(respt)
clr='rgb'
for i,kp in enumerate(k):
    plt.plot(zs,res[:,i],clr[i]+'-',label='k=%3.1fh/Mpc'%(kp/hub))
    plt.plot(zs,respt[:,i],clr[i]+'--')

plt.ylim(1e-2,1e2)
plt.xlabel('z')
plt.ylabel('SNR per mode')
plt.legend()
plt.semilogy()
plt.show()


           
