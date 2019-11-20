"""PUMA Noise simulator

Follows https://arxiv.org/abs/1810.09572.

Includes a general RadioTelescope class that defines a telescope in terms
of various dish, packing, and instrumental noise properties, as well as
instances of this class for the full and petite configurations of PUMA
(see https://arxiv.org/abs/1907.12559).

All spatial units are Mpc, not Mpc/h !!
"""

import numpy as np
from castorina import castorinaBias,castorinaPn
import pyccl as ccl


class RadioTelescope:
    """Class for computing signal and noise properties of a radio telescope.

    Uses signal and noise models from Appendices B and D of the Cosmic
    Visions 21cm white paper, https://arxiv.org/pdf/1810.09572v3.pdf.

    Attributes
    ----------
    C : ccl.Cosmology class
        CCL class defining the background cosmology.
    Nside : int, optional
        Number of receivers per side of square array (default: 256)
    D : float, optional
        Physical diameter of dishes, in m (default: 6)
    tint : float, optional
        Integration time of survey, in y (default: 5)
    fsky : float, optional
        Observed sky fraction (default: 0.5)
    effic : float, optional
        Dish aperture efficiency factor, such that the effective
        dish area is A_eff = effic * A_phys (default: 0.7)
    Tampl : float, optional
        Amplifier noise temperature, in K (default: 50)
    Tground : float, optional
        Ground temperature, in K (default: 300)
    omtcoupling : float, optional
        Optical efficiency of receivers, which boosts the effective
        Tampl by 1/omtcoupling (default: 0.9)
    skycoupling : float, optional
        Coupling of the primary beam to the sky, such that a fraction
        (1-skycoupling) of the beam hits the ground instead of the sky
        (default: 0.9)
    hexpack : bool, optional
        True if dishes are hex-packed, False if they are square-packed
        (default: True)
    """
    def __init__ (self,C,Nside=256, D=6, tint=5, fsky=0.5, effic=0.7, Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True):
        # CCL cosmology class
        self.C=C
        # Number of dishes per array side
        self.Nside=Nside
        # Total number of dishes
        self.Nd=Nside**2
        # Roughly, maximum baseline length in square array
        self.Dmax=Nside*np.sqrt(2)*D # m
        # Physical dish diameter
        self.D=D # m
        # Effective dish diameter
        self.Deff=self.D*np.sqrt(effic) # m
        # Total integration time
        self.ttotal=tint*365*24*3600 # s
        # Sky area
        self.Sarea=4*np.pi*fsky # sr
        # Sky fraction
        self.fsky=fsky
        # Effective dish area
        self.Ae=np.pi/4*D**2*effic # m^2
        # Contribution to system temperature from amplifier and groundspill
        # (Eq. D1 in paper)
        self.Tscope=Tampl/omtcoupling/skycoupling+Tground*(1-skycoupling)/skycoupling # K
        # Hex packing setting
        self.hexpack=hexpack

    def nofl(self,x):
        """Number density of baselines on the ground.

        Parameters
        ----------
        x : float or array
            Baseline length(s), in m.

        Returns
        -------
        res : float or array
            Number density of baselines of given length(s), in m^-2.
        """
        ### quadratic packing
        if (not self.hexpack):
            ### square packing
            a,b,B,C,D=0.4847, -0.330,  1.3157,  1.5975,  6.8390
        else:
            ### hexagonal packing
            a,b,B,C,D=0.56981864, -0.52741196,  0.8358006 ,  1.66354748,  7.31776875

        # Scale physical distances by Nside*D
        xn=np.asarray(x)/(self.Nside*self.D)
        # Fitting function prefactor
        n0=(self.Nside/self.D)**2 # m^-2
        # Fitting formula evaluation
        res=np.asarray( n0*(a+b*xn)/(1+B*xn**C)*np.exp(-(xn)**D) ) # m^-2
        # Impose numerical floor on result
        if (res.shape == ()):
            res = np.max([res,1e-10])
        else:
            res[res<1e-10]=1e-10

        return res


    def PNoise(self,z,kperp):
        """Thermal noise power spectrum.

        Parameters
        ----------
        z : float
            Redshift.
        kperp : float or array
            kperp value(s), in Mpc^-1.

        Returns
        -------
        Pn : float or array
            Thermal noise power spectrum, in K^2 Mpc^3.
        """

        # Observed wavelength
        lam=0.21*(1+z) # m
        # Comoving radial distance to redshift z
        r=ccl.comoving_radial_distance(self.C,1/(1.+z)) # Mpc
        # Conversion between kperp and uv-plane (vector norm) u
        u=np.asarray(kperp)*r/(2*np.pi)
        # Baseline length corresponding to u
        l=u*lam # m
        # Number density of baselines in uv plane
        Nu = self.nofl(l)*lam**2

        # Inaccurate approximation for uv-plane baseline density
        #umax=self.Dmax/lam
        #Nu=self.Nd**2/(2*np.pi*umax**2)

        # Field of view of single dish
        FOV=(lam/self.Deff)**2 # sr
        # Hubble parameter H(z)
        Hz=self.C['H0']*ccl.h_over_h0(self.C,1./(1.+z)) # km s^-1 Mpc^-1
        # Conversion factor from frequency to physical space
        y=3e5*(1+z)**2/(1420e6*Hz) # Mpc s

        # System temperature (sum of telescope and sky temperatures)
        Tsys=self.Tsky(1420./(1+z))+self.Tscope # K

        # 21cm noise power spectrum (Eq. D4 of paper).
        # Hard-codes 2 polarizations
        Pn=Tsys**2*r**2*y*(lam**4/self.Ae**2)* 1/(2*Nu*self.ttotal) * (self.Sarea/FOV) # K^2 Mpc^3

        # Catastrophically fail if we've gotten negative power spectrum values
        if np.any(Pn<0):
            print (Nu,Pn,l, self.nofl(l), self.nofl(l/2))
            stop()

        return Pn

    def PNoiseShot(self,z,Tb):
        """21cm shot noise power spectrum.

        Parameters
        ----------
        z : float
            Redshift.
        Tb : float
            Mean 21cm brightness temperature (your choice of units).

        Returns
        -------
        pn : float or array
            Shot noise power spectrum, in Mpc^3 times square of input Tb units.
        """
        return Tb**2*castorinaPn(z)/(self.C['h'])**3

    def PNoiseKFull(self,z,kperp,kpar, Tb=None,kparcut=0.01*0.7):
        """Full 21cm noise power spectrum, with specified kpar cut

        Parameters
        ----------
        z : float
            Redshift.
        kperp : array[nkpar,nkperp]
            2d array where columns are kperp values (in Mpc^-1) and rows are identical.
            Generate e.g. with np.outer(np.ones(nkpar),kperp_vec) where kperp_vec
            is a list of kperp values.
        kpar : array[nkpar,nkperp]
            2d array where rows are kpar values (in Mpc^-1) and columns are identical.
        Tb : float, optional
            Mean 21cm brightness temperature, in K (default: computed automatically).
        kparcut : float, optional
            Set Pnoise to large value if kpar<kparcut, in Mpc^-1 (default: 0.007).

        Returns
        -------
        Pn : array[nkpar,nkperp]
            Array of sums of 21cm thermal noise and shot noise power spectra, in K^2 Mpc^3.
        """
        assert(len(kperp.shape)==2)
        assert(len(kpar.shape)==2)
        if Tb is None:
            Tb=self.Tb(z)
        Pn=self.PNoise(z,kperp)+self.PNoiseShot(z,Tb)
        Pn[kpar<kparcut]=1e30
        return Pn

    def bias(self,z):
        """HI bias with redshift.

        Parameters
        ----------
        z : float or array
            Redshift(s).

        Returns
        -------
        b : float or array
            b_HI(z) values.
        """
        return castorinaBias(z)

    def Tsky(self,f):
        """Mean sky temperature, including Galactic synchrotron and CMB.

        Parameters
        ----------
        f : float or array
            Frequency or array of frequencies, in MHz.

        Returns
        -------
        Tsky : float or array
            Sky temperature(s), in K.
        """
        #return (f/100.)**(-2.4)*2000+2.7 ## from CVFisher
        return 25.*(np.asarray(f)/400.)**(-2.75) +2.75

    def TbTZ(self,z):
        """Approximation for mean 21cm brightness temperature.

        This is from Chang et al. 2008, https://arxiv.org/pdf/0709.3672.pdf,
        Eq. 1.

        Parameters
        ----------
        z : float or array
            Redshift(s).

        Returns
        -------
        Tb : float or array
            Temperature values, in K.
        """
        OmegaM=0.31
        z = np.asarray(z)
        return 0.3e-3*np.sqrt((1+z)/(2.5)*0.29/(OmegaM+(1.-OmegaM)/(1+z)**3))


    def Tb(self,z):
        """Approximation for mean 21cm brightness temperature.

        This is reasonably up-to-date, and comes from Eq. B1
        in the CV 21cm paper.

        Parameters
        ----------
        z : float or array
            Redshift(s).

        Returns
        -------
        Tb : float or array
            Temperature value(s), in K.
        """
        z = np.asarray(z)
        Ez=ccl.h_over_h0(self.C,1./(1.+z))
        # Note potentially misleading notation:
        # Ohi = (comoving density at z) / (critical density at z=0)
        Ohi=4e-4*(1+z)**0.6
        Tb=188e-3*self.C['h']/Ez*Ohi*(1+z)**2
        return Tb



    def cutWedge(self, noise, kperp, kpar, z, NW=3.0):
        """Cut the foreground wedge from a 2d noise power spectrum.

        Parameters
        ----------
        noise : array[nkpar,nkperp]
            2d noise power spectrum.
        kperp : array[nkpar,nkperp]
            2d array where columns are kperp values (in Mpc^-1) and rows are identical.
        kpar : array[nkpar,nkperp]
            2d array where rows are kpar values (in Mpc^-1) and columns are identical.
        z : float
            Redshift.
        NW : float, optional
            Multiplier defining wedge in terms of primary beam.
            (default = 3)

        Returns
        -------
        Pn : array[nkpar,nkperp]
            2d noise power spectrum where modes within wedge have noise set to
            large value.
        """
        # Comoving radial distance to redshift z
        r=ccl.comoving_radial_distance(self.C,1/(1.+z)) # Mpc
        # Hubble parameter H(z)
        H=self.C['H0']*ccl.h_over_h0(self.C,1./(1.+z)) # km s^-1 Mpc^-1
        # Slope that defines wedge as kpar < kperp * slope.
        # See Eq. C1 from the CV 21cm paper.
        slope= r*H/3e5 * 1.22 *0.21/self.D * NW / 2.0 # dimensionless

        # Boost noise for modes within wedge
        noiseout=np.copy(noise)
        noiseout[np.where(kpar<kperp*slope)]=1e30

        return noiseout


    def PSSensitivityTransit (self, freq=600, bandwidth=900):
        """One sigma point source transit sensitivity

        Also prints some quantities for comparison: Tsys, t_eff
        for the input telescope and CHIME, and the point source
        sensitivity for CHIME.

        Parameters
        ----------
        freq : float, optional
            Frequency, in MHz (default = 600).
        bandwidth : float, optional
            Bandwidth, in MHz (default = 900).

        Returns
        -------
        onesigma : float
            Point source sensitivity, in Jy.
        """
        # Boltzmann constant
        kB=1.38064852e-23 # J K^-1
        # Observed wavelength
        lam = 3e8/(freq*1e6) # m
        # Total instrument collecting area
        Acoll= self.Ae*self.Nd # m^2
        # Dish field of view
        FOV=(lam/self.Deff)**2 # m^2

        # Effective transit times for specified telescope and CHIME (both in s)
        teff=np.sqrt(FOV)/(2*np.pi*np.cos(30/180*np.pi))*24*3600 ## 30 deg south
        teffchime=(lam/20)/(2*np.pi*np.cos(50/180*np.pi))*24*3600 ## 50 deg north
        print ("Acoll*np.sqrt(teff*bandwidth*1e6)",Acoll * np.sqrt(2*teff*bandwidth*1e6))

        # System temperature
        Tsys = self.Tsky(freq)+self.Tscope # K

        # One sigma sensitivity
        onesigma= 2 * kB * Tsys /  ( Acoll * np.sqrt(2*teff*bandwidth*1e6)) / 1e-26 ## to Jy
        print ("Tsys",Tsys)
        print ("teff=",teff,teffchime)
        print ("CHIME:", 10* 2 *kB * Tsys / (0.7*80*100*np.sqrt(2*teffchime * 400e6))/1e-26)
        return onesigma



class PUMA(RadioTelescope):
    """Specs for full PUMA telescope (see https://arxiv.org/pdf/1907.12559.pdf).

    Survey time is given as 5/4 years to account for 0.5 filling factor.
    """
    def __init__ (self,C):
        RadioTelescope.__init__(self,C,Nside=256, D=6, tint=5/4, fsky=0.5, effic=0.7,
                                Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True)

class PUMAPetite(RadioTelescope):
    """Specs for petite PUMA telescope (see https://arxiv.org/pdf/1907.12559.pdf).

    Survey time is given as 5/4 years to account for 0.5 filling factor.
    """
    def __init__ (self,C):
        RadioTelescope.__init__(self,C,Nside=100, D=6, tint=5/4, fsky=0.5, effic=0.7,
                                Tampl=50., Tground=300., omtcoupling=0.9, skycoupling=0.9, hexpack=True)
