import numpy as np
#from cobaya.conventions import _path_install
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.likelihoods._base_classes import _InstallableLikelihood
import os,sys
from .utils import *
from .fg import *
# import utils
# import fg
from soapack import interfaces as sints
from pkg_resources import resource_filename

dfroot = resource_filename("bplike","data/actpolfull_dr4.01/data/data_act/")

dfroot_coadd_w = resource_filename("bplike","data/bplike_data/big_coadd_weights/200226/")
dfroot_coadd_d = resource_filename("bplike","data/bplike_data/coadd_data/")
dfroot_fg = resource_filename("bplike","data/actpolfull_dr4.01/data/Fg/")
dfroot_bpass = resource_filename("bplike","data/bplike_data/bpass/")

sz_temp_file = dfroot_fg+"cl_tsz_150_bat.dat"
sz_x_cib_temp_file = dfroot_fg+"sz_x_cib_template.dat"
ksz_temp_file = dfroot_fg+"cl_ksz_bat.dat"

def get_band(array):
    a = array.split("_")[1]
    assert a[0]=='f'
    if a[1:]=='150':
        return '150'
    elif a[1:]=='090':
        return '95'
    else:
        raise ValueError

def save_coadd_matrix(spec,band1,band2,flux,path_root):
    import pandas as pd

    from scipy.linalg import block_diag
    if flux=='15mJy':
        regions = ['deep56']
    elif flux=='100mJy':
        regions = ['boss'] + [f'advact_window{x}' for x in range(6)]
    else:
        raise ValueError

    nbin = 59
    rbin = 7 # remove first 7 bins
    def rmap(r):
        if r[:6]=='advact': return 'advact'
        else: return r

    barrays = []
    icovs = []
    for region in regions:
        order = np.load(f"{dfroot_coadd_w}{region}_all_C_ell_data_order_190918.npy")
        df = pd.DataFrame(order,columns=['t1','t2','region','s1','s2','a1','a2']).stack().str.decode('utf-8').unstack()
        df = df[(df.t1==spec[0]) & (df.t2==spec[1]) & (df.region==rmap(region)+"_")]
        arrays = []
        for index, row in df.iterrows():
            b1 = get_band(row.a1)
            b2 = get_band(row.a2)
            if (b1==band1 and b2==band2) or (b1==band2 and b2==band1):
                arrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                barrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))

        adf = pd.DataFrame(arrays,columns = ['i','r','s1','s2','a1','a2'])
        ids = adf.i.to_numpy()
        oids = []
        for ind in ids:
            oids = oids + list(range(ind*nbin+rbin,(ind+1)*nbin))

        """
        Covmat selection
        """
        cov = np.load(dfroot_coadd_w+f"{region}_all_covmat_190918.npy")
        ocov = cov[oids,:][:,oids]
        icovs.append(np.linalg.inv(ocov))

    icov = block_diag(*icovs)
    nspec = 1
    nbins = 52
    norig = len(barrays)
    N_spec = nbins*nspec
    # building projection matrix with shape nbins*nspec x nbins*norig
    pmat = np.identity(N_spec)
    Pmat = np.identity(N_spec)
    for i in range(1,norig):
        Pmat = np.append(Pmat,pmat,axis=1)
    icov_ibin = np.linalg.inv(np.dot(Pmat,np.dot(icov,Pmat.T)))
    np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov.txt',icov)
    np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov_ibin.txt',icov_ibin)
    np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_pmat.txt',Pmat)
    barrays = np.array(barrays,dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')])
    np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_arrays.txt', barrays,fmt=['%d','%s','%s','%s','%s','%s'])


def load_coadd_matrix(spec,band1,band2,flux,path_root):
    icov = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov.txt',)
    icov_ibin = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov_ibin.txt',)
    pmat = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_pmat.txt',)
    arrays = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_arrays.txt',dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')],ndmin=1)
    return icov,icov_ibin,pmat,arrays


class StevePower(object):
    def __init__(self,froot,flux,infval=1e10,tt_lmin=600,tt_lmax=None):
        spec=np.loadtxt(f"{froot}coadd_cl_{flux}_data_200124.txt")
        cov =np.loadtxt(f'{froot}coadd_cov_{flux}_200519.txt')
        self.bbl =np.loadtxt(f'{froot}coadd_bpwf_{flux}_191127_lmin2.txt').reshape((10,52,7924))
        self.spec = spec[:520]
        self.cov = cov[:520,:520]
        nbin = 52
        self.ells = np.arange(2,7924+2)
        rells = np.repeat(self.ells[None],10,axis=0)
        self.ls = self.bin(rells)

        if tt_lmin is not None:
            n = 3
            ids = []
            ids = np.argwhere(self.ls<tt_lmin)[:,0]
            ids = ids[ids<nbin*3]
            self.cov[:,ids] = 0
            self.cov[ids,:] = 0
            self.cov[ids,ids] = infval

        if tt_lmax is not None:
            n = 3
            ids = []
            ids = np.argwhere(self.ls>tt_lmax)[:,0]
            ids = ids[ids<nbin*3]
            self.cov[:,ids] = 0
            self.cov[ids,:] = 0
            self.cov[ids,ids] = infval

        self.cinv = np.linalg.inv(self.cov)

    def bin(self,dls):
        bdl = np.einsum('...k,...k',self.bbl,dls[:,None,:])
        return bdl.reshape(-1)

    def select(self,bls,spec,band1,band2,shift=52):
        I = {'tt':0,'te':3,'ee':7}
        i = { 'tt':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2},
              'te':{('95','95'): 0,('95','150'): 1,('150','95'): 2,('150','150'): 3},
              'ee':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2} }
        mind = i[spec][(band1,band2)]
        sel = np.s_[(I[spec]+mind)*shift:(I[spec]+mind+1)*shift]
        if bls.ndim==1: return bls[sel]
        elif bls.ndim==2: return bls[sel,sel]
        else: raise ValueError

class act_pylike_extended(_InstallableLikelihood):

    def initialize(self):

        self.l_max = 6000

        self.log.info("Initialising.")
        # Load path_params from yaml file
        self.fparams = config_from_yaml('params_extended.yml')['fixed']
        self.aparams = config_from_yaml('params_extended.yml')['act_like']
        self.bpmodes = config_from_yaml('params_extended.yml')['bpass_modes']
        self.bands = self.aparams['bands']

        # Read data
        self.prepare_data()

        # State requisites to the theory code
        self.requested_cls = ["tt", "te", "ee"]
        self.expected_params = [
            "a_tsz", # tSZ
            "xi", # tSZ-CIB cross-correlation coefficient
            "a_c", # clustered CIB power
            "beta_CIB", # CIB frequency scaling
            "a_ksz", # kSZ
            "a_d", # dusty/CIB Poisson
            "a_p_tt_15", # TT radio Poisson with given flux cut
            "a_p_tt_100", # TT radio Poisson with given flux cut
            "a_p_te", # TE Poisson sources
            "a_p_ee", # EE Poisson sources
            "a_g_tt", # TT Galactic dust at ell=500
            "a_g_te", # TE Galactic dust at ell=500
            "a_g_ee", # EE Galactic dust at ell=500
            "a_s_te", # TE Synchrotron at ell=500
            "a_s_ee", # EE Synchrotron at ell=500
            "cal_95",
            "cal_150",
            "yp_95",
            "yp_150"
        ]

        self.cal_params = []
        nbands = len(self.bands)
        for i in range(nbands):
            self.cal_params.append(f"ct{i}") # Temperature Calibration
            self.cal_params.append(f"yp{i}") # Polarization gain


        self.log.debug(
            f"ACT-like {self.flux} initialized." )

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params,
            name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    def get_requirements(self):
        return {'Cl': {'tt': self.l_max,'te': self.l_max,'ee': self.l_max}}

    def logp(self, **params_values):
        # return 0
        cl = self.theory.get_Cl(ell_factor=True)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        ps_vec = self._get_power_spectra(cl, **params_values)
        delta = self.sp.spec - ps_vec
        logp = -0.5 * np.dot(delta,np.dot(self.sp.cinv,delta))
        self.log.debug(
            f"ACT-like {self.flux} lnLike value = {logp} (chisquare = {-2 * logp})")
        return logp

    def prepare_data(self, verbose=False):
        flux = self.flux
        # self.sp = StevePower("data/actpol_2f_full_s1316_2flux_fin/data/data_act/ps_200519/",self.flux)

        self.sp = StevePower(dfroot,self.flux)
        if self.bandpass:
            sbands = { 'TT':[('95','95'),('95','150'),('150','150')],
                       'TE':[('95','95'),('95','150'),('150','95'),('150','150')],
                       'EE':[('95','95'),('95','150'),('150','150')] }
            self.coadd_data = {}
            for spec in ['TT','TE','EE']:
                self.coadd_data[spec] = {}
                for bands in sbands[spec]:
                    band1,band2 = bands
                    self.coadd_data[spec][bands] = load_coadd_matrix(spec,band1,band2,
                                                                     self.flux,f"{dfroot_coadd_d}coadds_20200305")

            dm = sints.ACTmr3()
            beam_dict = {}
            bp_dict = {}
            cfreq_dict = {}
            cfreqs = {'pa1_f150':148.9,'pa2_f150':149.1,'pa3_f150':146.6,'pa3_f090':97.1}

            if flux=='15mJy':
                anames = [f'd56_0{i}' for i in range(1,7)]
            elif flux=='100mJy':
                anames = [f'boss_0{i}' for i in range(1,5)] +  [f's16_0{i}' for i in range(1,4)]
            else:
                raise ValueError

            pnames = []
            for aname in anames:
                season,array,freq,patch = sints.arrays(aname,'season'),sints.arrays(aname,'array'),sints.arrays(aname,'freq'),sints.arrays(aname,'region')
                pname = '_'.join([season,array,freq])
                pnames.append(pname)
                beam_dict[pname] = dm.get_beam_fname(season,patch,array+"_"+freq, version=None)
                bp_dict[pname] = dfroot_bpass+dm.get_bandpass_file_name(array+"_"+freq)
                cfreq_dict[pname] = cfreqs[array + "_" + freq]
        else:
            pnames = None
            bp_dict = None
            beam_dict = None
            cfreq_dict = None

        self.fgpower = ForegroundPowers(self.fparams,self.sp.ells,
                                            sz_temp_file,ksz_temp_file,sz_x_cib_temp_file,flux_cut=self.flux,
                                            arrays=pnames,bp_file_dict=bp_dict,beam_file_dict=beam_dict,cfreq_dict=cfreq_dict)

    def _get_power_spectra(self, cl, **params_values):

        if self.theory_debug is not None:
            ells,cltt,clee,clte = np.loadtxt(self.theory_debug,usecols=[0,1,2,4],unpack=True)
            assert ells[0] == 2
            assert ells[1] == 3
            cl = {}
            cl['ell'] = np.zeros(2+self.l_max+50)
            cl['tt'] = np.zeros(2+self.l_max+50)
            cl['te'] = np.zeros(2+self.l_max+50)
            cl['ee'] = np.zeros(2+self.l_max+50)
            cl['ell'][1] = 1
            cl['ell'][2:] = ells[:self.l_max+50]
            cl['tt'][2:] = cltt[:self.l_max+50]
            cl['te'][2:] = clte[:self.l_max+50]
            cl['ee'][2:] = clee[:self.l_max+50]


        fgdict =    {k: params_values[k] for k in self.expected_params}
        fgdict.update(self.fparams)
        nells_camb = cl['ell'].size
        nells = self.sp.ells.size
        assert cl['ell'][0]==0
        assert cl['ell'][1]==1
        assert self.sp.ells[0]==2
        assert self.sp.ells[1]==3
        ptt = np.zeros(nells+2)
        pte = np.zeros(nells+2)
        pee = np.zeros(nells+2)
        ptt[2:nells_camb] = cl['tt'][2:]
        pte[2:nells_camb] = cl['te'][2:]
        pee[2:nells_camb] = cl['ee'][2:]

        if self.bandpass:
            fpower = self.fgpower.get_theory_bandpassed(self.coadd_data,self.sp.ells,
                                                        self.sp.bbl,ptt[2:],pte[2:],pee[2:],fgdict,lmax=self.l_max)
        else:
            fpower = self.fgpower.get_theory(self.sp.ells,self.sp.bin,ptt[2:],pte[2:],pee[2:],fgdict,lmax=self.l_max)
        return fpower



class act15(act_pylike_extended):
    flux = '15mJy'

class act100(act_pylike_extended):
    flux = '100mJy'
