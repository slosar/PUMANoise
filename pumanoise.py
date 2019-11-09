import numpy as np
from castorina import castorinaBias,castorinaPn
import pyccl as ccl

##
## PUMA Noise simulator
## Follows https://arxiv.org/abs/1810.09572
##
## All units are Mpc, not Mpc/h !!
##

class RadioTelescope:
    def __init__ (self,C,Nside=256, D=6, tint=5, fsky=0.5, effic=0.7, Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True):
        self.C=C
        self.Nside=Nside
        self.Nd=Nside**2
        self.Dmax=Nside*np.sqrt(2)*D
        self.D=D
        self.Deff=self.D*np.sqrt(effic)
        self.ttotal=tint*365*24*3600
        self.Sarea=4*np.pi*fsky
        self.fsky=fsky
        self.Ae=np.pi/4*D**2*effic
        self.Tscope=Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling
        self.hexpack=hexpack

    def nofl(self,x):
        """ Returns baseline density  """
        ### quadratic packing
        if (not self.hexpack):
            ### square packing
            a,b,B,C,D=0.4847, -0.330,  1.3157,  1.5975,  6.8390
        else:
            ### hexagonal packing
            a,b,B,C,D=0.56981864, -0.52741196,  0.8358006 ,  1.66354748,  7.31776875
        xn=x/(self.Nside*self.D)
        n0=(self.Nside/self.D)**2
        res=np.array(n0*(a+b*xn)/(1+B*xn**C)*np.exp(-(xn)**D))
        if (type(res)==np.ndarray):
            res[res<1e-10]=1e-10
        # if (res<1e-10):
        #     res=1e-10
        return res


    def PNoise(self,z,kperp):
        """ Thermal noise power in Mpc^3 """
        lam=0.21*(1+z)
        r=ccl.comoving_radial_distance(self.C,1/(1.+z))
        u=kperp*r/(2*np.pi)
        l=u*lam
        Nu = self.nofl(l)*lam**2
        #umax=self.Dmax/lam
        #Nu=self.Nd**2/(2*np.pi*umax**2)
        FOV=(lam/self.Deff)**2
        Hz=self.C['H0']*ccl.h_over_h0(self.C,1./(1.+z))
        y=3e5*(1+z)**2/(1420e6*Hz)
        Tsys=self.Tsky(1420./(1+z))+self.Tscope
        Pn=Tsys**2*r**2*y*(lam**4/self.Ae**2)* 1/(2*Nu*self.ttotal) * (self.Sarea/FOV)
        if np.any(Pn<0):
            print (Nu,Pn,l, self.nofl(l), self.nofl(l/2))
            stop()
        return Pn

    def PNoiseShot(self,z,Tb):
        return Tb**2*castorinaPn(z)/(self.C['h'])**3

    def PNoiseKFull(self,z,kperp,kpar, Tb=None,kparcut=0.01*0.7):
        """" Full noise power in Mpc^3, including shotnoise and cuts"""
        assert(len(kperp.shape)==2)
        assert(len(kpar.shape)==2)
        if Tb is None:
            Tb=self.Tb(z)
        Pn=self.PNoise(z,kperp)+self.PNoiseShot(z,Tb)
        Pn[kpar<kparcut]=1e30
        return Pn

    def bias(self,z):
        return castorinaBias(z)

    def Tsky(self,f):
        #return (f/100.)**(-2.4)*2000+2.7 ## from CVFisher
        return 25.*(f/400.)**(-2.75) +2.75

    def TbTZ(self,z):
        OmegaM=0.31
        return 0.3e-3*np.sqrt((1+z)/(2.5)*0.29/(OmegaM+(1.-OmegaM)/(1+z)**3))


    def Tb(self,z):
        Ez=ccl.h_over_h0(self.C,1./(1.+z))
        # Note potentially misleading notation:
        # Ohi = (comoving density at z) / (critical density at z=0)
        Ohi=4e-4*(1+z)**0.6
        Tb=188e-3*self.C['h']/Ez*Ohi*(1+z)**2
        return Tb



    def cutWedge(self, noise, kperp, kpar, z, NW=3.0):
        r=ccl.comoving_radial_distance(self.C,1/(1.+z))
        H=self.C['H0']*ccl.h_over_h0(self.C,1./(1.+z))
        slope= r*H/3e5 * 1.22 *0.21/self.D * NW / 2.0
        noiseout=np.copy(noise)
        noiseout[np.where(kpar<kperp*slope)]=1e30
        return noiseout


    def PSSensitivityTransit (self, freq=600, bandwidth=900):
        kB=1.38064852e-23
        lam = 3e8/(freq*1e6)
        Acoll= self.Ae*self.Nd
        FOV=(lam/self.Deff)**2
        teff=np.sqrt(FOV)/(2*np.pi*np.cos(30/180*np.pi))*24*3600 ## 30 deg south
        teffchime=(lam/20)/(2*np.pi*np.cos(50/180*np.pi))*24*3600 ## 50 deg north
        print ("Acoll*np.sqrt(teff*bandwidth*1e6)",Acoll * np.sqrt(2*teff*bandwidth*1e6))

        Tsys = self.Tsky(freq)+self.Tscope
        onesigma= 2 * kB * Tsys /  ( Acoll * np.sqrt(2*teff*bandwidth*1e6)) / 1e-26 ## to Jy
        print ("Tsys",Tsys)
        print ("teff=",teff,teffchime)
        print ("CHIME:", 10* 2 *kB * Tsys / (0.7*80*100*np.sqrt(2*teffchime * 400e6))/1e-26)
        return onesigma



class PUMA(RadioTelescope):
    def __init__ (self,C):
        RadioTelescope.__init__(self,C,Nside=256, D=6, tint=5/4, fsky=0.5, effic=0.7,
                                Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True)

class PUMAPetite(RadioTelescope):
    def __init__ (self,C):
        RadioTelescope.__init__(self,C,Nside=100, D=6, tint=5/4, fsky=0.5, effic=0.7,
                                Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True)
