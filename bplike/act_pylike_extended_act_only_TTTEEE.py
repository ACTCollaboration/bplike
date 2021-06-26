import numpy as np
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.likelihoods.base_classes import InstallableLikelihood
import os,sys
from .config import *
from .utils import *
from .fg import *
from .stevepower import *
from soapack import interfaces as sints
from pkg_resources import resource_filename

import multiprocessing
import functools



class act_pylike_extended_act_only_TTTEEE(InstallableLikelihood):


    def initialize(self):

        self.log.info("Initialising.")
        self.expected_params = [
            "a_tsz", # tSZ
            "xi", # tSZ-CIB cross-correlation coefficient
            "a_c", # clustered CIB power
            "beta_CIB", # CIB frequency scaling
            "beta_radio", # radio frequency scaling
            "a_ksz", # kSZ
            "a_d", # dusty/CIB Poisson
            "a_p_tt_15", # TT radio Poisson with given flux cut
            "a_p_tt_100", # TT radio Poisson with given flux cut
            "a_p_te", # TE Poisson sources
            "a_p_ee", # EE Poisson sources
            "a_g_tt_15", # TT Galactic dust at ell=500
            "a_g_tt_100", # TT Galactic dust at ell=500
            "a_g_te_15", # TE Galactic dust at ell=500
            "a_g_te_100", # TE Galactic dust at ell=500
            "a_g_ee_15", # EE Galactic dust at ell=500
            "a_g_ee_100", # EE Galactic dust at ell=500
            "a_s_te", # TE Synchrotron at ell=500
            "a_s_ee"] # EE Synchrotron at ell=500

        self.cal_yp_act_only =[
            "cal_95",
            "cal_150",
            "leak_150",
            "leak_95",
            "yp_95",
            "yp_150"]
        self.cal_yp_act_plus_planck =[
              "cal_090",
              # "yp_090",
              "cal_100",
              # "yp_100",
              "cal_143",
              # "yp_143",
              "cal_150",
              # "yp_150",
              "cal_217",
              # "yp_217",
              "cal_353",
              # "yp_353",
              "cal_545"#,
              # "yp_545"
              ]

        if self.use_act_planck == 'no':
            # self.l_max = 6000
            self.l_max = 6051
            self.fparams = config_from_yaml('params.yml')['fixed']
            self.aparams = config_from_yaml('params.yml')['act_like']
            self.bpmodes = config_from_yaml('params.yml')['bpass_modes']
            # for the act only lkl:

            cal_yp =  self.cal_yp_act_only



        self.expected_params = list(np.concatenate((self.expected_params,cal_yp)))
        # print('expected params: ',self.expected_params)
        # exit(0)
        self.bands = self.aparams['bands']


        # Read data
        # print('preparing data')
        self.prepare_data()


        self.cal_params = []
        nbands = len(self.bands)
        # print('bands: ', self.bands)
        for i in range(nbands):
            self.cal_params.append(f"ct{i}") # Temperature Calibration
            self.cal_params.append(f"yp{i}") # Polarization gain


        self.log.debug(
            f"ACT-like {self.flux} initialized." )

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        # print('params: ', self.use_act_planck)
        # print('input params: ',self.input_params)
        if self.use_act_planck == 'yes':
            l = self.input_params
            l_pop_cal_yp = [s for s in self.cal_yp_act_only  if s not in self.cal_yp_act_plus_planck]
            new_l = [s for s in l if s not in l_pop_cal_yp ]
            self.input_params = new_l
        elif self.use_act_planck == 'no':
            l = self.input_params
            l_pop_cal_yp = [s for s in self.cal_yp_act_plus_planck  if s not in self.cal_yp_act_only]
            new_l = [s for s in l if s not in l_pop_cal_yp ]
            self.input_params = new_l

    def get_requirements(self):
        l_max = 5000
        # l_max = self.l_max
        if self.use_act_planck == 'no':
            # reqs = {'Cl': {'tt': self.l_max}}
            reqs = {'Cl': {'tt': l_max,'te': l_max,'ee': l_max}}
        elif self.use_act_planck == 'yes':
            reqs = {'Cl': {'tt': l_max}}
        return reqs

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=True)
        # print("cl's: ",cl)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        comps = ['tot','primary','tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']


        # print('loglike getting power spectra')
        ps = self._get_power_spectra(cl, lkl_setup = self, **params_values)
        # print('loglike done getting power spectra')
        ps_vec = ps['tot']

        n_bins = self.sp.n_bins
        if self.bandpass:
            bps = '_bp_'
        else:
            bps = '_'

        fac = self.sp.ls*(self.sp.ls+1.)/2./np.pi

        if self.use_act_planck == 'yes':

            # print('ps_vec : ', ps_vec[:n_bins]/fac[:n_bins])
            # print('self.sp.spec : ', self.sp.spec[:n_bins])
            dls_theory = ps_vec # Primary + FG

            ls_theory = self.sp.ls
            delta = self.sp.spec - dls_theory/fac
            if self.save_theory_data_spectra:
                np.save(path_to_output+'/ls_theory_'+self.flux+bps+'act_planck_'+self.root_theory_data_spectra+'.npy',ls_theory)

                for comp in comps:
                    np.save(path_to_output+'/dls_theory_'+comp+'_'+self.flux+bps+'act_planck_'+self.root_theory_data_spectra+'.npy',ps[comp])



        elif self.use_act_planck == 'no':
            dls_theory = ps_vec.copy()
            ls_theory = self.sp.ls

            # add T to P leakage Eq. 26-27 of choi et al https://arxiv.org/pdf/2007.07289.pdf
            # T95E95
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'T095'))[:,0]
            dls_theory[ids_leak_TE] = dls_theory[ids_leak_TE] + dls_theory[ids_leak_TT]*params_values['leak_95']*self.sp.leak_95[:]

            # T95E150
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'T150'))[:,0]
            dls_theory[ids_leak_TE] = dls_theory[ids_leak_TE] + dls_theory[ids_leak_TT]*params_values['leak_150']*self.sp.leak_150[:]

            # T150E95
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'T150'))[:,0]
            dls_theory[ids_leak_TE] = dls_theory[ids_leak_TE] + dls_theory[ids_leak_TT]*params_values['leak_95']*self.sp.leak_95[:]

            # T150E150
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'T150'))[:,0]
            dls_theory[ids_leak_TE] = dls_theory[ids_leak_TE] + dls_theory[ids_leak_TT]*params_values['leak_150']*self.sp.leak_150[:]

            # E095E095
            ids_leak_EE = np.argwhere((self.sp.rfband1 == 'E095') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_ET = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'T095'))[:,0]
            dls_theory[ids_leak_EE] = dls_theory[ids_leak_EE] \
                                    + dls_theory[ids_leak_TE]*params_values['leak_95']*self.sp.leak_95[:]\
                                    + dls_theory[ids_leak_ET]*params_values['leak_95']*self.sp.leak_95[:]\
                                    + dls_theory[ids_leak_TT]*params_values['leak_95']*self.sp.leak_95[:]*params_values['leak_95']*self.sp.leak_95[:]

            # E150E150
            ids_leak_EE = np.argwhere((self.sp.rfband1 == 'E150') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_ET = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'T150'))[:,0]
            dls_theory[ids_leak_EE] = dls_theory[ids_leak_EE] \
                                    + dls_theory[ids_leak_TE]*params_values['leak_150']*self.sp.leak_150[:]\
                                    + dls_theory[ids_leak_ET]*params_values['leak_150']*self.sp.leak_150[:]\
                                    + dls_theory[ids_leak_TT]*params_values['leak_150']*self.sp.leak_150[:]*params_values['leak_150']*self.sp.leak_150[:]


            # E095E150
            ids_leak_EE = np.argwhere((self.sp.rfband1 == 'E095') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_TE = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'E150'))[:,0]
            ids_leak_ET = np.argwhere((self.sp.rfband1 == 'T150') & (self.sp.rfband2 == 'E095'))[:,0]
            ids_leak_TT = np.argwhere((self.sp.rfband1 == 'T095') & (self.sp.rfband2 == 'T150'))[:,0]
            dls_theory[ids_leak_EE] = dls_theory[ids_leak_EE]\
                                    + dls_theory[ids_leak_TE]*params_values['leak_95']*self.sp.leak_95[:]\
                                    + dls_theory[ids_leak_ET]*params_values['leak_150']*self.sp.leak_150[:]\
                                    + dls_theory[ids_leak_TT]*params_values['leak_95']*self.sp.leak_95[:]*params_values['leak_150']*self.sp.leak_150[:]


            delta = self.sp.spec - dls_theory
            if self.save_theory_data_spectra:
                np.save(path_to_output+'/ls_theory_'+self.flux+bps+'act_only_'+self.root_theory_data_spectra+'.npy',ls_theory)
                for comp in comps:
                    np.save(path_to_output+'/dls_theory_'+comp+'_'+self.flux+bps+'act_only_'+self.root_theory_data_spectra+'.npy',ps[comp]*fac)


        logp = -0.5 * np.dot(delta,np.dot(self.sp.cinv,delta))
        if self.theory_debug is not None:
            print('[debug] logp: ',logp)
        self.log.debug(
            f"ACT-like {self.flux} lnLike value = {logp} (chisquare = {-2 * logp})")
        return logp

    def prepare_data(self, verbose=False):
        str_current = '[bplike prepare_data] '
        flux = self.flux
        data_root = dfroot
        print(str_current+'Collecting power spectra from %s and with flux %s'%(data_root,self.flux))
        self.sp = StevePower(data_root,self.flux,l_max_data = self.l_max_data)
        if self.bandpass:
            # print('doing bandpasses')
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
            # print('anames:',anames)

            pnames = []
            # print('loop over anames:',anames)
            for aname in anames:
                season,array,freq,patch = sints.arrays(aname,'season'),sints.arrays(aname,'array'),sints.arrays(aname,'freq'),sints.arrays(aname,'region')
                pname = '_'.join([season,array,freq])
                pnames.append(pname)
                beam_dict[pname] = dm.get_beam_fname(season,patch,array+"_"+freq, version=None)
                bp_dict[pname] = dfroot_bpass+dm.get_bandpass_file_name(array+"_"+freq)
                cfreq_dict[pname] = cfreqs[array + "_" + freq]

        else:
            print(str_current+'Not using bandpass - set in the param file if you want to include these.')
            pnames = None
            bp_dict = None
            beam_dict = None
            cfreq_dict = None

        self.fgpower = ForegroundPowers(self.fparams,self.sp.ells,
                                            cib_temp_file,
                                            sz_temp_file,
                                            ksz_temp_file,
                                            sz_x_cib_temp_file,
                                            flux_cut=self.flux,
                                            arrays=pnames,
                                            bp_file_dict=bp_dict,
                                            beam_file_dict=beam_dict,
                                            cfreq_dict=cfreq_dict,
                                            lkl_setup = self)


    def _get_power_spectra(self, cl, lkl_setup = None, **params_values):
        # print('getting power spectra')
        l_min = 2
        # print('cls:',cl)

        if self.theory_debug is not None:
            print('[debug] theory debug:',self.theory_debug)
            # ells,cltt,clee,clte = np.loadtxt(self.theory_debug,usecols=[0,1,2,4],unpack=True) # mat's debig
            ells,cltt,clte,clee = np.loadtxt(self.theory_debug,usecols=[0,1,2,3],unpack=True) # boris's debug

            assert ells[0] == 2
            assert ells[1] == 3
            cl = {}
            l_max = len(ells) + 2

            cl['ell'] = np.zeros(l_max)
            cl['tt'] = np.zeros(l_max)
            cl['te'] = np.zeros(l_max)
            cl['ee'] = np.zeros(l_max)

            cl['ell'][1] = 1
            cl['ell'][l_min:] = ells[:l_max]
            cl['tt'][l_min:] = cltt[:l_max]
            cl['te'][l_min:] = clte[:l_max]
            cl['ee'][l_min:] = clee[:l_max]


        fgdict =    {k: params_values[k] for k in self.expected_params}
        fgdict.update(self.fparams)
        nells_camb = cl['ell'].size

        nells = self.sp.ells.size

        assert cl['ell'][0]==0
        assert cl['ell'][1]==1
        assert self.sp.ells[0]==l_min
        assert self.sp.ells[1]==l_min + 1
        ptt = np.zeros(nells+l_min)
        pte = np.zeros(nells+l_min)
        pee = np.zeros(nells+l_min)

        ptt[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
        pte[l_min:nells_camb] = cl['te'][l_min:len(ptt[l_min:nells_camb])+l_min]
        pee[l_min:nells_camb] = cl['ee'][l_min:len(ptt[l_min:nells_camb])+l_min]

        # print('starting get theory')

        comps = ['primary','tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']

        if self.bandpass:
            # print('doing theory bandpass')
            fpower = {}
            if self.theory_debug is not None:
                import time
                start = time.time()
                print('[debug] comp = ', 'tot')
            fpower['tot'] = self.fgpower.get_theory_bandpassed(self.coadd_data,
                                                        self.sp.ells,
                                                        self.sp.bbl,
                                                        ptt[l_min:],
                                                        pte[l_min:],
                                                        pee[l_min:],
                                                        fgdict,
                                                        lmax=self.l_max,
                                                        lkl_setup = lkl_setup)
            if self.save_theory_data_spectra:
                for comp in comps:
                    print('computing ',comp)
                    fpower[comp] = self.fgpower.get_theory_bandpassed_comp(self.coadd_data,
                                                            self.sp.ells,
                                                            self.sp.bbl,
                                                            ptt[l_min:],
                                                            pte[l_min:],
                                                            pee[l_min:],
                                                            fgdict,
                                                            lmax=self.l_max,
                                                            lkl_setup = lkl_setup,
                                                            comp = comp)

            return fpower


        else:
            fpower = {}
            if self.theory_debug is not None:
                import time
                start = time.time()
                print('[debug] comp = ', 'tot')
            fpower['tot'] = self.fgpower.get_theory(self.sp.ells,
                                             self.sp.bin,
                                             ptt[l_min:],
                                             pte[l_min:],
                                             pee[l_min:],
                                             fgdict,
                                             lmax=self.l_max,
                                             lkl_setup = lkl_setup)
            # print('fpower:',fpower)


            if self.theory_debug is not None:
                print('[debug] time for tot: ', time.time() - start)
            if self.save_theory_data_spectra:
                for comp in comps:
                    print('computing ',comp)
                    if self.theory_debug is not None:
                        print('[debug] comp = ', comp)
                        start = time.time()
                    fpower[comp] = self.fgpower.get_comp(self.sp.ells,
                                                     self.sp.bin,
                                                     ptt[l_min:],
                                                     pte[l_min:],
                                                     pee[l_min:],
                                                     fgdict,
                                                     lmax=self.l_max,
                                                     lkl_setup = lkl_setup,
                                                     comp = comp)
                    if self.theory_debug is not None:
                        print('[debug] time for comp: ', time.time() - start)

            return fpower



class act15_act_only_TTTEEE(act_pylike_extended_act_only_TTTEEE):
    flux = '15mJy'
    use_act_planck = 'no'

class act100_act_only_TTTEEE(act_pylike_extended_act_only_TTTEEE):
    flux = '100mJy'
    use_act_planck = 'no'
