#!/usr/bin/env python
from pumanoise import PUMA, PUMAPetite
import matplotlib.pyplot as plt
import pyccl as ccl
import numpy as np
from matplotlib.colors import LogNorm
from scipy.integrate import quad

### This cross-checks agains Martin White's code

C=ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)
puma=PUMA(C)

def thermal_n(kperp,zz,D=6.0,Ns=256,hex=True):

    """The thermal noise for PUMA -- note noise rescaling from 5->5/4 yr."""
    # Some constants.
    etaA = 0.7 # Aperture efficiency.
    Aeff = etaA*np.pi*(D/2)**2 # m^2
    lam21= 0.21*(1+zz) # m
    nuobs= 1420/(1+zz) # MHz
    # The cosmology-dependent factors.
    hub  = C['h'] 
    Ez   = ccl.h_over_h0(C,1./(1.+zz))
    chi  = ccl.comoving_radial_distance(C,1/(1.+zz)) * hub         # Mpc/h.

    #hub = cc.H(0).value / 100.0
    #Ez = cc.H(zz).value / cc.H(0).value
    #chi = cc.comoving_distance(zz).value * hub # Mpc/h.
    OmHI = 4e-4*(1+zz)**0.6 / Ez**2
    Tbar = 0.188*hub*(1+zz)**2*Ez*OmHI # K
    # Eq. (3.3) of Chen++19
    d2V = chi**2*2997.925/Ez*(1+zz)**2
    # Eq. (3.5) of Chen++19
    if hex: # Hexagonal array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.5698,-0.5274,0.8358,1.6635,7.3177
        uu = kperp*chi/(2*np.pi)
        xx = uu*lam21/Ns/D # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        #nbase[uu< D/lam21 ]=1e-30
        nbase[uu>Ns*D/lam21*1.3]=1e-30
    else: # Square array of Ns^2 elements.
        n0,c1,c2,c3,c4,c5 = (Ns/D)**2,0.4847,-0.33,1.3157,1.5974,6.8390
        uu = kperp*chi/(2*np.pi)
        xx = uu*lam21/Ns/D # Dimensionless.
        nbase= n0*(c1+c2*xx)/(1+c3*xx**c4)*np.exp(-xx**c5) * lam21**2 + 1e-30
        #nbase[uu< D/lam21 ]=1e-30
        nbase[uu>Ns*D/lam21*1.4]=1e-30
        # Eq. (3.2) of Chen++19
    npol = 2
    fsky = 0.5
    tobs = 5.*365.25*24.*3600. # sec.
    tobs/= 4.0 # Scale to 1/2-filled array.
    Tamp = 62.0 # K
    Tgnd = 33.0 # K
    Tsky = 2.7 + 25*(400./nuobs)**2.75 # K
    Tsys = Tamp + Tsky + Tgnd
    Omp = (lam21/D)**2/etaA
    # Return Pth in "cosmological units", with the Tbar divided out.
    Pth = (Tsys/Tbar)**2*(lam21**2/Aeff)**2 *\
          4*np.pi*fsky/Omp/(npol*1420e6*tobs*nbase) * d2V
    return(Pth)



kperp=np.logspace(np.log10(1e-2),np.log10(1),100)
hub  = C['h'] 

for i,zz in enumerate([2.,4.,6.]):
    mwnoise=thermal_n(kperp,zz)
    asnoise=puma.PNoise(zz,kperp*hub)/puma.Tb(zz)**2 *hub**3
    plt.plot(kperp, asnoise,'rgb'[i]+'-',label='AS z=%i'%zz)
    plt.plot(kperp, mwnoise,'rgb'[i]+'--',label='MW z=%i'%zz)

plt.xlabel('k [h/Mpc]')
plt.ylabel('Pn [Mpc/h]^3')
plt.legend()
plt.semilogy()
plt.show()
