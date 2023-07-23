import numpy as np
#from cobaya.conventions import _path_install
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.likelihoods.base_classes import InstallableLikelihood
import os,sys
from .config import *
from .utils import *
from .fg import *
from .stevepower import *
# import utils
# import fg
from soapack import interfaces as sints
from pkg_resources import resource_filename

import multiprocessing
import functools
from scipy.interpolate import interp1d


class act_pylike_extended_act_TT_plus_planck_TT(InstallableLikelihood):


    def initialize(self):

        self.log.info("Initialising.")
        if self.use_multiple_tsz_bpw == 'yes':
            # print('multiple tsz bpw')
            # print(self.params)
            # # self.params.pop('a_tsz')
            # print(self.params)
            # exit(0)
            self.expected_params = [
                "a_tsz_3000", # tSZ
                "a_tsz_2500",
                # Amplitude of tSZ @ l=2000
                "a_tsz_2000",
                # Amplitude of tSZ @ l=1500
                "a_tsz_1500",
                # Amplitude of tSZ @ l=1000
                "a_tsz_1000",
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
        else:
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
        # self.l_max = 3924
        self.l_max = self.l_max_data
        # self.l_max = 6051
        self.fparams = config_from_yaml('params_extended.yml')['fixed']
        self.aparams = config_from_yaml('params_extended.yml')['act_like']
        self.bpmodes = config_from_yaml('params_extended.yml')['bpass_modes']
        # for the act+planck lkl:

        cal_yp = self.cal_yp_act_plus_planck





        self.expected_params = list(np.concatenate((self.expected_params,cal_yp)))
        self.bands = self.aparams['bands']

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
        l = self.input_params
        l_pop_cal_yp = [s for s in self.cal_yp_act_only  if s not in self.cal_yp_act_plus_planck]
        new_l = [s for s in l if s not in l_pop_cal_yp ]
        self.input_params = new_l

    def get_requirements(self):
        l_max = 8000
        if self.use_classy_sz_tsz == 'no':
            reqs = {'Cl': {'tt': l_max}}
        elif self.use_classy_sz_tsz == 'yes':
            reqs = {'Cl': {'tt': l_max},'Cl_sz':{}}
        return reqs

    def logp(self, **params_values):
        # return 0
        # print('doing logp')

        cl = self.theory.get_Cl(ell_factor=True)
        cl_sz = None
        if self.use_classy_sz_tsz == 'yes':
            theory = self.theory.get_Cl_sz()
            cl_1h_theory = theory['1h']
            cl_2h_theory = theory['2h']
            cl_sz = np.asarray(list(cl_1h_theory)) + np.asarray(list(cl_2h_theory))
            cl_sz_ells = theory['ell']
            cl_sz = (cl_sz_ells,cl_sz)
        return self.loglike(cl,cl_sz, **params_values)

    def loglike(self, cl,cl_sz, **params_values):
        comps = ['tot','primary','tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']
        ps = self._get_power_spectra(cl,cl_sz=cl_sz, lkl_setup = self, **params_values)
        ps_vec = ps['tot']
        n_bins = self.sp.n_bins
        if self.bandpass:
            bps = '_bp_'
        else:
            bps = '_'

        fac = self.sp.ls*(self.sp.ls+1.)/2./np.pi
        dls_theory = ps_vec # Primary + FG

        ls_theory = self.sp.ls
        delta = self.sp.spec - dls_theory/fac

        # uncomment below to print residuals
        # label_bps = []
        # for b1,b2 in zip(self.sp.fband1,self.sp.fband2):
        #     label_bps.append(b1 +'x' +b2)
        # ps_list = ['090x090', '090x100', '090x143', '090x150', '090x217', '090x353', '090x545', '100x100', '100x143', '143x143', '100x150', '143x150', '150x150', '150x217', '150x353', '150x545', '100x217', '143x217', '217x217', '100x353', '143x353', '217x353', '353x353', '100x545', '143x545', '217x545', '353x545', '545x545']
        # for ps in ps_list:
        #     j = label_bps.index(ps)
        #     # print(j)
        #     # exit(0)
        #     for l,cl_data,cl_theory,delta_cl,sigmas_cl in zip(ls_theory[j*self.sp.n_bins:(j+1)*self.sp.n_bins],\
        #     (dls_theory/fac)[j*self.sp.n_bins:(j+1)*self.sp.n_bins],\
        #     self.sp.spec[j*self.sp.n_bins:(j+1)*self.sp.n_bins],\
        #     delta[j*self.sp.n_bins:(j+1)*self.sp.n_bins],\
        #     np.diagonal(self.sp.cov)[j*self.sp.n_bins:(j+1)*self.sp.n_bins]):
        #         print("%s : l = %.3e\t cl_data = %.3e\t cl_theo = %.3e\t dcl = %.3e\t sigma = %.3e"%(ps,l,cl_data,cl_theory,delta_cl,np.sqrt(sigmas_cl)))
        #     print("\n\n")
        #     print("###################################################")
        #     print("########################\t%s\t#########################"%self.flux)
        #     print("###################################################")
        #     print("\n\n")

        if self.save_theory_data_spectra:
            np.save(path_to_output+'/ls_theory_'+self.flux+bps+'act_planck_'+self.root_theory_data_spectra+'.npy',ls_theory)

            for comp in comps:
                # print(ps['cibc'])
                np.save(path_to_output+'/dls_theory_'+comp+'_'+self.flux+bps+'act_planck_'+self.root_theory_data_spectra+'.npy',ps[comp])

        logp = -0.5 * np.dot(delta,np.dot(self.sp.cinv,delta))
        if self.theory_debug is not None:
            print('[debug] logp: ',logp)
        self.log.debug(
            f"ACT-like {self.flux} lnLike value = {logp} (chisquare = {-2 * logp})")
        # print(f"ACT-like {self.flux} lnLike value = {logp} (chisquare = {-2 * logp})")
        return logp

    def prepare_data(self, verbose=False):
        str_current = '[bplike prepare_data] '
        flux = self.flux
        if self.l_max_data == 7924:
            data_root = path_to_data + '/act_planck_data_260422/'
            # data_root = path_to_data + '/act_planck_data_210610/'
        else:
            data_root = path_to_data + '/act_planck_data_210328/'
        self.sp = StevePower_extended(data_root,self.flux,l_max_data = self.l_max_data, diag_cov_only = self.diag_cov_only)
    # exit(0)
        if self.bandpass:
            self.coadd_data = {}
            for spec in ['TT']:
                self.coadd_data[spec] = {}
                for i in range(self.sp.n_specs):
                    band1 = self.sp.fband1[i]
                    band2 = self.sp.fband2[i]
                    bands = (band1,band2)

                    if flux=='15mJy':
                        reg = 'deep56'
                    elif flux=='100mJy':
                        reg = 'boss'
                    self.coadd_data[spec][bands] = load_coadd_matrix(spec,band1,band2,
                                                              self.flux,f"{data_root}{reg}")
            dm = sints.ACTmr3() # data model ACT

            beam_dict = {}
            bp_dict = {}
            cfreq_dict = {}
            cfreqs = {'pa1_f150':148.9,
                      'pa2_f150':149.1,
                      'pa3_f150':146.6,
                      'pa3_f090':97.1,
                      'pa0_f100':100.1,
                      'pa0_f143':143.1,
                      'pa0_f217':217.1,
                      'pa0_f353':353.1,
                      'pa0_f545':545.1,
                      }

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

            # data_root = path_to_data + '/act_planck_data_210328/'
            bp_dict['s12_pa0_f100'] = data_root + 'HFI_BANDPASS_F100_reformat.txt'
            pnames.append('s12_pa0_f100')
            bp_dict['s12_pa0_f143'] = data_root + 'HFI_BANDPASS_F143_reformat.txt'
            pnames.append('s12_pa0_f143')
            bp_dict['s12_pa0_f217'] = data_root + 'HFI_BANDPASS_F217_reformat.txt'
            pnames.append('s12_pa0_f217')
            bp_dict['s12_pa0_f353'] = data_root + 'HFI_BANDPASS_F353_reformat.txt'
            pnames.append('s12_pa0_f353')
            bp_dict['s12_pa0_f545'] = data_root + 'HFI_BANDPASS_F545_reformat.txt'
            pnames.append('s12_pa0_f545')


            beam_dict['s12_pa0_f100'] = data_root + 'HFI_BEAM_resave_210414_F100.txt'
            beam_dict['s12_pa0_f143'] = data_root + 'HFI_BEAM_resave_210414_F143.txt'
            beam_dict['s12_pa0_f217'] = data_root + 'HFI_BEAM_resave_210414_F217.txt'
            beam_dict['s12_pa0_f353'] = data_root + 'HFI_BEAM_resave_210414_F353.txt'
            beam_dict['s12_pa0_f545'] = data_root + 'HFI_BEAM_resave_210414_F545.txt'

            cfreq_dict['s12_pa0_f100'] =cfreqs['pa0_f100']
            cfreq_dict['s12_pa0_f143'] =cfreqs['pa0_f143']
            cfreq_dict['s12_pa0_f217'] =cfreqs['pa0_f217']
            cfreq_dict['s12_pa0_f353'] =cfreqs['pa0_f353']
            cfreq_dict['s12_pa0_f545'] =cfreqs['pa0_f545']

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


    def _get_power_spectra(self, cl, cl_sz = None, lkl_setup = None, **params_values):
        # print('getting power spectra')
        l_min = 2
        # print('cls:',cl)

        if self.theory_debug is not None:
            print('[debug] theory debug:',self.theory_debug)
            # ells,cltt,clee,clte = np.loadtxt(self.theory_debug,usecols=[0,1,2,4],unpack=True) # mat's debig
            ells,cltt,clte,clee = np.loadtxt(path_to_data+'/bf_ACTPol_lcdm.minimum.theory_cl',usecols=[0,1,2,3],unpack=True)
            # print('ell0,ell1:',ells[0],ells[1],ells[:self.l_max])

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
        ptsz = None

        ptt[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
        pte[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
        pee[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
        # ptsz = blbla
        if cl_sz is not None:
            ls = cl_sz[0]
            pow = cl_sz[1]
            powfunc = interp1d(ls,pow)
            ptsz = powfunc(self.sp.ells)

        comps = ['primary','tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']

        if self.bandpass:
            # print('doing theory bandpass')
            fpower = {}
            if self.theory_debug is not None:
                import time
                start = time.time()
                print('[debug] comp = ', 'tot')
            # print('[debug] comp = ', 'tot')
            fpower['tot'] = self.fgpower.get_theory_bandpassed(self.coadd_data,
                                                        self.sp.ells,
                                                        self.sp.bbl,
                                                        ptt[l_min:],
                                                        pte[l_min:],
                                                        pee[l_min:],
                                                        fgdict,
                                                        lmax=self.l_max,
                                                        lkl_setup = lkl_setup,
                                                        ptsz = ptsz)
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
                                                            comp = comp,
                                                            ptsz = ptsz)
            if self.theory_debug is not None:
                print('[debug] time for tot: ', time.time() - start)
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


            if self.theory_debug is not None:
                print('[debug] time for tot: ', time.time() - start)
            if self.save_theory_data_spectra:
                for comp in comps:
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
                                                     comp = comp,
                                                     ptsz = ptsz)
                    if self.theory_debug is not None:
                        print('[debug] time for comp: ', time.time() - start)

            return fpower


class act15_act_TT_plus_planck_TT(act_pylike_extended_act_TT_plus_planck_TT):
    flux = '15mJy'
    use_act_planck = 'yes'

class act100_act_TT_plus_planck_TT(act_pylike_extended_act_TT_plus_planck_TT):
    flux = '100mJy'
    use_act_planck = 'yes'
