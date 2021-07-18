import numpy as np
from scipy.interpolate import interp1d
import tilec
from tilec.fg import ArraySED


def get_template(ells,template_file,ell_pivot=None):
    ls,pow = np.loadtxt(template_file,unpack=True)
    powfunc = interp1d(ls,pow)
    if ell_pivot is not None:
        pow_pivot = powfunc(ell_pivot)
    else:
        pow_pivot = 1.
    return powfunc(ells)/pow_pivot


class ForegroundPowers(ArraySED):
    def __init__(self,params,ells,
                 cib_temp_file,
                 sz_temp_file,
                 ksz_temp_file,
                 sz_x_cib_temp_file,
                 flux_cut,
                 arrays=None,
                 bp_file_dict=None,
                 beam_file_dict=None,
                 cfreq_dict=None,
                 lkl_setup = None,
                 comps = ['tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']
             ):
        # print('getting ells')
        self.ells= ells
        self.cib_temp = get_template(ells,cib_temp_file,ell_pivot=3000)
        self.tsz_temp = get_template(ells,sz_temp_file,ell_pivot=params['high_ell0'])
        self.ksz_temp = get_template(ells,ksz_temp_file,ell_pivot=params['high_ell0'])
        self.tsz_x_cib_temp = get_template(ells,sz_x_cib_temp_file,ell_pivot=None)
        self.fcut = flux_cut
        self.lkl_setup  = lkl_setup

        self.effs = {}
        if lkl_setup.use_act_planck == 'no' or lkl_setup == None:
            self.effs['95'] = {}
            self.effs['150'] = {}
            for c in ['tsz','dust','syn']:
                self.effs['95'][c] = params[f'f{c}_95_{self.fcut}']
                self.effs['150'][c] = params[f'f{c}_150_{self.fcut}']
        elif lkl_setup.use_act_planck == 'yes':
            for cfreqs in lkl_setup.sp.cfreqs_list:
                self.effs[cfreqs] = {}
                for c in ['tsz','dust','syn']:
                    self.effs[cfreqs][c] = params[f'f{c}_{cfreqs}_{self.fcut}']

        self.comps = comps
        ArraySED.__init__(self,arrays=arrays,bp_file_dict=bp_file_dict,beam_file_dict=beam_file_dict,cfreq_dict=cfreq_dict)


    def get_component_scale_dependence(self,comp,param_dict):
        p = param_dict
        comp = comp.lower()
        if comp == 'tsz':
            return self.tsz_temp
        elif comp == 'ksz':
            return self.ksz_temp
        elif comp=='tsz_x_cib':
            return self.tsz_x_cib_temp
        elif comp=='cibc':
            return self.cib_temp
            # return self.ells * (self.ells+1) / p['high_ell0'] / (p['high_ell0']+1.) * (self.ells / p['high_ell0'])**p['cibc_n']
        elif comp=='poisson':
            return self.ells * (self.ells+1) / p['high_ell0'] / (p['high_ell0']+1.)
        elif comp in ['galdust_t','galdust_p','galsync_t','galsync_p']:
            return (self.ells/p['low_ell0'])**p[f'{comp}_n']

        else:
            raise ValueError


    def get_power(self,spec,comps,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None,lmax=None,ptsz=None):
        ocomps = [comp.lower() for comp in comps]
        spec = spec.lower()
        tpow = 0
        if spec=='tt':
            if ('tsz' in ocomps) or ('tsz_x_cib' in ocomps):
                e1tsz = eff_freq_ghz1['tsz'] if eff_freq_ghz1 is not None else None
                e2tsz = eff_freq_ghz2['tsz'] if eff_freq_ghz2 is not None else None
                f1_tsz = self.get_response("tSZ",array=array1,norm_freq_ghz=params['nu0'],
                                           eff_freq_ghz=e1tsz,params=params,lmax=lmax)
                f2_tsz = self.get_response("tSZ",array=array2,norm_freq_ghz=params['nu0'],
                                           eff_freq_ghz=e2tsz,params=params,lmax=lmax)
                if (f2_tsz.size<lmax and f2_tsz.size>1):
                    new_f2 = np.zeros(lmax)
                    new_f2[:f2_tsz.size] = f2_tsz
                    new_f2[f2_tsz.size:] = np.repeat(f2_tsz[f2_tsz.size-1], lmax-f2_tsz.size)
                    f2_tsz = new_f2
                if (f1_tsz.size<lmax and f1_tsz.size>1):
                    new_f1 = np.zeros(lmax)
                    new_f1[:f1_tsz.size] = f1_tsz
                    new_f1[f1_tsz.size:] = np.repeat(f1_tsz[f1_tsz.size-1], lmax-f1_tsz.size)
                    f1_tsz = new_f1

            if ('cibc' in ocomps) or ('cibp' in ocomps) or ('tsz_x_cib' in ocomps):
                e1dusty = eff_freq_ghz1['dust'] if eff_freq_ghz1 is not None else None
                e2dusty = eff_freq_ghz2['dust'] if eff_freq_ghz2 is not None else None
                f1_cib = self.get_response("CIB",array=array1,norm_freq_ghz=params['nu0'],
                                           eff_freq_ghz=e1dusty,params=params,lmax=lmax)
                f2_cib = self.get_response("CIB",array=array2,norm_freq_ghz=params['nu0'],
                                           eff_freq_ghz=e2dusty,params=params,lmax=lmax)
                if (f2_cib.size<lmax and f2_cib.size>1):
                    new_f2 = np.zeros(lmax)
                    new_f2[:f2_cib.size] = f2_cib
                    f2_cib = new_f2
                    new_f2[f2_cib.size:] = np.repeat(f2_cib[f2_cib.size-1], lmax-f2_cib.size)
                if (f1_cib.size<lmax and f1_cib.size>1):
                    new_f1 = np.zeros(lmax)
                    new_f1[:f1_cib.size] = f1_cib
                    new_f1[f1_cib.size:] = np.repeat(f1_cib[f1_cib.size-1], lmax-f1_cib.size)
                    f1_cib = new_f1
            if ('tsz' in ocomps):
                if ptsz is not None:
                    tpow = tpow + f1_tsz *f2_tsz *ptsz
                else:
                    if self.lkl_setup.use_multiple_tsz_bpw == 'yes':
                        # print(params['a_tsz'])
                        # print(params['a_tsz_3000'])
                        # print(params['a_tsz_2500'])
                        # print(params['a_tsz_2000'])
                        # print(params['a_tsz_1500'])
                        # print(params['a_tsz_1000'])
                        # self.get_component_scale_dependence('tSZ',params)
                        l_full = np.arange(2,np.shape(self.get_component_scale_dependence('tSZ',params))[0]+2)
                        l_1000 = np.where(l_full>=1147.20269044)
                        # print(l_full[l_1000])
                        l_1500 = np.where((l_full<1147.20269044) | (l_full >=1509.80364848))
                        # print(l_full[l_1500])
                        l_2000 = np.where((l_full<1509.80364848) | (l_full >=1987.01334642))
                        # print(l_full[l_2000])
                        l_2500 = np.where((l_full<1987.01334642) | (l_full >=2615.05662862))
                        # print(l_full[l_2500])
                        l_3000 = np.where((l_full<2615.05662862))
                        full_tsz_temp = self.get_component_scale_dependence('tSZ',params).copy()
                        tsz_temp_1000 = full_tsz_temp.copy()
                        tsz_temp_1500 = full_tsz_temp.copy()
                        tsz_temp_2000 = full_tsz_temp.copy()
                        tsz_temp_2500 = full_tsz_temp.copy()
                        tsz_temp_3000 = full_tsz_temp.copy()
                        tsz_temp_1000[l_1000] = 0
                        tsz_temp_1500[l_1500] = 0
                        tsz_temp_2000[l_2000] = 0
                        tsz_temp_2500[l_2500] = 0
                        tsz_temp_3000[l_3000] = 0
                        # exit(0)
                        tpow = tpow \
                        + f1_tsz *f2_tsz *params['a_tsz_3000']*tsz_temp_3000\
                        + f1_tsz *f2_tsz *params['a_tsz_2500']*tsz_temp_2500\
                        + f1_tsz *f2_tsz *params['a_tsz_2000']*tsz_temp_2000\
                        + f1_tsz *f2_tsz *params['a_tsz_1500']*tsz_temp_1500\
                        + f1_tsz *f2_tsz *params['a_tsz_1000']*tsz_temp_1000
                        # exit(0)
                    else:
                        tpow = tpow + f1_tsz *f2_tsz *params['a_tsz']*self.get_component_scale_dependence('tSZ',params)
            if ('cibc' in ocomps):
                tpow = tpow + f1_cib*f2_cib*params['a_c']*self.get_component_scale_dependence('cibc',params)
            if ('cibp' in ocomps):
                tpow = tpow + f1_cib*f2_cib*params['a_d']*self.get_component_scale_dependence('poisson',params)
            if ('ksz' in ocomps):
                tpow = tpow + params['a_ksz']*self.get_component_scale_dependence('kSZ',params)
            if ('tsz_x_cib' in ocomps):
                if self.lkl_setup.use_multiple_tsz_bpw == 'yes':
                    l_full = np.arange(2,np.shape(self.get_component_scale_dependence('tSZ',params))[0]+2)
                    l_1000 = np.where(l_full>=1147.20269044)
                    # print(l_full[l_1000])
                    l_1500 = np.where((l_full<1147.20269044) | (l_full >=1509.80364848))
                    # print(l_full[l_1500])
                    l_2000 = np.where((l_full<1509.80364848) | (l_full >=1987.01334642))
                    # print(l_full[l_2000])
                    l_2500 = np.where((l_full<1987.01334642) | (l_full >=2615.05662862))
                    # print(l_full[l_2500])
                    l_3000 = np.where((l_full<2615.05662862))
                    full_tsz_temp = self.get_component_scale_dependence('tSZ_x_cib',params).copy()
                    tsz_temp_1000 = full_tsz_temp.copy()
                    tsz_temp_1500 = full_tsz_temp.copy()
                    tsz_temp_2000 = full_tsz_temp.copy()
                    tsz_temp_2500 = full_tsz_temp.copy()
                    tsz_temp_3000 = full_tsz_temp.copy()
                    tsz_temp_1000[l_1000] = 0
                    tsz_temp_1500[l_1500] = 0
                    tsz_temp_2000[l_2000] = 0
                    tsz_temp_2500[l_2500] = 0
                    tsz_temp_3000[l_3000] = 0

                    a_c = params['a_c']
                    a_sz_3000 = params['a_tsz_3000']
                    a_sz_2500 = params['a_tsz_2500']
                    a_sz_2000 = params['a_tsz_2000']
                    a_sz_1500 = params['a_tsz_1500']
                    a_sz_1000 = params['a_tsz_1000']
                    xi = params['xi']
                    fp = (f1_tsz*f2_cib + f2_tsz*f1_cib)/2.
                    tpow = tpow \
                    - 2.*fp*xi*np.sqrt(a_sz_3000*a_c)*tsz_temp_3000 \
                    - 2.*fp*xi*np.sqrt(a_sz_2500*a_c)*tsz_temp_2500 \
                    - 2.*fp*xi*np.sqrt(a_sz_2000*a_c)*tsz_temp_2000 \
                    - 2.*fp*xi*np.sqrt(a_sz_1500*a_c)*tsz_temp_1500 \
                    - 2.*fp*xi*np.sqrt(a_sz_1000*a_c)*tsz_temp_1000
                else:
                    a_c = params['a_c']
                    a_sz = params['a_tsz']
                    xi = params['xi']
                    fp = (f1_tsz*f2_cib + f2_tsz*f1_cib)/2.
                    tpow = tpow - 2.*fp*xi*np.sqrt(a_sz*a_c)*self.get_component_scale_dependence('tsz_x_cib',params)
        if 'radio' in ocomps:
            e1syn = eff_freq_ghz1['syn'] if eff_freq_ghz1 is not None else None
            e2syn = eff_freq_ghz2['syn'] if eff_freq_ghz2 is not None else None
            f1 = self.get_response("radio",
                                    array=array1,
                                    norm_freq_ghz=params['nu0'],
                                    eff_freq_ghz=e1syn,
                                    params=params,
                                    lmax=lmax)
            f2 = self.get_response("radio",
                                    array=array2,
                                    norm_freq_ghz=params['nu0'],
                                    eff_freq_ghz=e2syn,
                                    params=params,
                                    lmax=lmax)
            if (f2.size<lmax and f2.size>1):
                new_f2 = np.zeros(lmax)
                new_f2[:f2.size] = f2
                new_f2[f2.size:] = np.repeat(f2[f2.size-1], lmax-f2.size)
                f2 = new_f2
            if (f1.size<lmax and f1.size>1):
                new_f1 = np.zeros(lmax)
                new_f1[:f1.size] = f1
                new_f1[f1.size:] = np.repeat(f1[f1.size-1], lmax-f1.size)
                f1 = new_f1
            if spec=='tt':
                if self.fcut == '15mJy':
                    fnum = 15
                elif self.fcut == '100mJy':
                    fnum = 100
                else:
                    raise ValueError
                rparam = f'a_p_{spec}_{fnum}'
            else:
                rparam = f'a_p_{spec}'
            tpow = tpow + f1*f2*params[rparam]*self.get_component_scale_dependence('poisson',params)

        if 'galdust' in ocomps:
            e1dusty = eff_freq_ghz1['dust'] if eff_freq_ghz1 is not None else None
            e2dusty = eff_freq_ghz2['dust'] if eff_freq_ghz2 is not None else None
            f1 = self.get_response("galdust",array=array1,
                                   norm_freq_ghz=params['nu0'],
                                   eff_freq_ghz=e1dusty,
                                   params=params,
                                   dust_beta_param_name='beta_galdust',#'beta_galdust',
                                   # dust_beta_param_name='beta_galdust',
                                   lmax=lmax)
            f2 = self.get_response("galdust",array=array2,
                                    norm_freq_ghz=params['nu0'],
                                   eff_freq_ghz=e2dusty,
                                   params=params,
                                   dust_beta_param_name='beta_galdust',#'beta_galdust',
                                   # dust_beta_param_name='beta_galdust',
                                   lmax=lmax)

            if (f2.size<lmax and f2.size>1):
                new_f2 = np.zeros(lmax)
                new_f2[:f2.size] = f2
                new_f2[f2.size:] = np.repeat(f2[f2.size-1], lmax-f2.size)
                f2 = new_f2
            if (f1.size<lmax and f1.size>1):
                new_f1 = np.zeros(lmax)
                new_f1[:f1.size] = f1
                new_f1[f1.size:] = np.repeat(f1[f1.size-1], lmax-f1.size)
                f1 = new_f1
            scale_str = 'galdust_t' if spec=='tt' else 'galdust_p'
            if self.fcut == '15mJy':
                fnum = 15
            elif self.fcut == '100mJy':
                fnum = 100
            else:
                raise ValueError
            rparam = f'a_g_{spec}_{fnum}'
            tpow = tpow + f1*f2*params[rparam]*self.get_component_scale_dependence(scale_str,params) # correct one
        if spec!='tt':
            if 'galsyn' in ocomps:
                e1syn = eff_freq_ghz1['syn'] if eff_freq_ghz1 is not None else None
                e2syn = eff_freq_ghz2['syn'] if eff_freq_ghz2 is not None else None
                f1 = self.get_response("radio",array=array1,norm_freq_ghz=params['nu0'],
                                          eff_freq_ghz=e1syn,params=params,radio_beta_param_name='beta_galsyn',lmax=lmax)
                f2 = self.get_response("radio",array=array2,norm_freq_ghz=params['nu0'],
                                          eff_freq_ghz=e2syn,params=params,radio_beta_param_name='beta_galsyn',lmax=lmax)
                scale_str = 'galsync_t' if spec=='tt' else 'galsync_p'
                tpow = tpow + f1*f2*params[f'a_s_{spec}']*self.get_component_scale_dependence(scale_str,params)

        return tpow

    def get_ksz_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['ksz'],params=params,
                         eff_freq_ghz1=None,array1=None,
                         eff_freq_ghz2=None,array2=None)

    def get_tsz_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['tsz'],params=params,
                              eff_freq_ghz1={'tsz':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'tsz':eff_freq_ghz2},array2=array2)

    def get_tsz_x_cib_power(self,spec,params,
                      eff_freq_ghz_sz1=None,
                      eff_freq_ghz_sz2=None,
                      eff_freq_ghz_cib1=None,
                      eff_freq_ghz_cib2=None,
                      array1=None,array2=None):
        return self.get_power(spec,['tsz_x_cib'],params=params,
                              eff_freq_ghz1={'tsz':eff_freq_ghz_sz1,'dust':eff_freq_ghz_cib1},array1=array1,
                              eff_freq_ghz2={'tsz':eff_freq_ghz_sz2,'dust':eff_freq_ghz_cib2},array2=array2)

    def get_cibc_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['cibc'],params=params,
                              eff_freq_ghz1={'dust':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'dust':eff_freq_ghz2},array2=array2)

    def get_cibp_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['cibp'],params=params,
                              eff_freq_ghz1={'dust':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'dust':eff_freq_ghz2},array2=array2)

    def get_radio_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['radio'],params=params,
                              eff_freq_ghz1={'syn':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'syn':eff_freq_ghz2},array2=array2)

    def get_galdust_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['galdust'],params=params,
                              eff_freq_ghz1={'dust':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'dust':eff_freq_ghz2},array2=array2)

    def get_galsync_power(self,spec,params,
                      eff_freq_ghz1=None,array1=None,
                      eff_freq_ghz2=None,array2=None):
        return self.get_power(spec,['galsyn'],params=params,
                              eff_freq_ghz1={'syn':eff_freq_ghz1},array1=array1,
                              eff_freq_ghz2={'syn':eff_freq_ghz2},array2=array2)


    def get_theory(self,ells,bin_func,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None,ptsz=None):
        # print('getting theory')
        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0

        if lkl_setup.use_act_planck == 'yes':
            # print('use_act_planck :', lkl_setup.use_act_planck)
            # exit(0)
            # dls = np.zeros((28,3924))
            dls = np.zeros((28,lkl_setup.sp.l_max))
            for i in range(28):
                band1 = lkl_setup.sp.fband1[i]
                band2 = lkl_setup.sp.fband2[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']
                # print('dltt 0-10',dltt[0:10])
                dls[i] = (dltt + self.get_power('TT',self.comps,params,
                                            eff_freq_ghz1=self.effs[band1],array1=None,
                                            eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2
                # print('dls 0-10',dls[0:10])
            return bin_func(dls/ells/(ells+1.)*2.*np.pi)
        else:
            dls = np.zeros((10,lkl_setup.sp.l_max))
            for i in range(10):
                if i<3:
                    band1 = {0:'95',1:'95',2:'150'}[i]
                    band2 = {0:'95',1:'150',2:'150'}[i]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    dls[i] = (dltt + self.get_power('TT',self.comps,params,
                                                eff_freq_ghz1=self.effs[band1],array1=None,
                                                eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2
                elif i>=3 and i<=6:
                    band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                    band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y = params[f'yp_{band2}']
                    dls[i] = (dlte + self.get_power('TE',self.comps,params,
                                                eff_freq_ghz1=self.effs[band1],array1=None,
                                                eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max) ) * c1 * c2 * y
                else:
                    band1 = {0:'95',1:'95',2:'150'}[i-7]
                    band2 = {0:'95',1:'150',2:'150'}[i-7]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y1 = params[f'yp_{band1}']
                    y2 = params[f'yp_{band2}']
                    dls[i] =  (dlee + self.get_power('EE',self.comps,params,
                                                eff_freq_ghz1=self.effs[band1],array1=None,
                                                eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max) ) * c1 * c2 * y1 * y2

            return bin_func(dls/ells/(ells+1.)*2.*np.pi)


    def get_primary(self,ells,bin_func,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None):
        # print('getting theory')
        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0

        if lkl_setup.use_act_planck == 'yes':
            # print('use_act_planck :', lkl_setup.use_act_planck)
            # dls = np.zeros((28,3924))
            dls = np.zeros((28,lkl_setup.sp.l_max))
            for i in range(28):
                band1 = lkl_setup.sp.fband1[i]
                band2 = lkl_setup.sp.fband2[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']
                # print('dltt 0-10',dltt[0:10])
                dls[i] = (dltt ) * c1 * c2
                # print('dls 0-10',dls[0:10])
            return bin_func(dls/ells/(ells+1.)*2.*np.pi)
        else:
            dls = np.zeros((10,lkl_setup.sp.l_max))
            for i in range(10):
                if i<3:
                    band1 = {0:'95',1:'95',2:'150'}[i]
                    band2 = {0:'95',1:'150',2:'150'}[i]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    # print('dltt 0-10',dltt[0:10])
                    dls[i] = (dltt ) * c1 * c2
                    # print('dls 0-10',dls[0:10])
                elif i>=3 and i<=6:
                    band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                    band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y = params[f'yp_{band2}']
                    dls[i] = (dlte  ) * c1 * c2 * y
                else:
                    band1 = {0:'95',1:'95',2:'150'}[i-7]
                    band2 = {0:'95',1:'150',2:'150'}[i-7]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y1 = params[f'yp_{band1}']
                    y2 = params[f'yp_{band2}']
                    dls[i] =  (dlee ) * c1 * c2 * y1 * y2

            return bin_func(dls/ells/(ells+1.)*2.*np.pi)



    def get_comp(self,ells,bin_func,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None, comp = None,ptsz=None):
        # print('getting theory')
        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0

        if lkl_setup.use_act_planck == 'yes':
            # print('use_act_planck :', lkl_setup.use_act_planck)
            # dls = np.zeros((28,3924))
            dls = np.zeros((28,lkl_setup.sp.l_max))
            for i in range(28):
                # band1 = {0:'090',1:'100',2:'143',3:'150',4:'217',5:'353',6:'545'}[i]
                # band2 = {0:'090',1:'100',2:'143',3:'150',4:'217',5:'353',6:'545'}[i]
                band1 = lkl_setup.sp.fband1[i]
                band2 = lkl_setup.sp.fband2[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']
                # print('dltt 0-10',dltt[0:10])
                # print('c1,c2:',c1,c2)
                # print('b1,b2:',band1,band2)
                # print('effs1,effs2:',self.effs[band1],self.effs[band2])
                # print('dltt 0-10',dltt[0:10])
                if comp == 'primary':
                    dls[i] = (dltt ) * c1 * c2
                else:
                    dls[i] = (self.get_power('TT',[comp],params,
                                                eff_freq_ghz1=self.effs[band1],array1=None,
                                                eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2

                # print('dls 0-10',dls[i][0:10])
            # exit(0)
            return bin_func(dls/ells/(ells+1.)*2.*np.pi)
        else:
            dls = np.zeros((10,lkl_setup.sp.l_max))
            for i in range(10):
                if i<3:
                    band1 = {0:'95',1:'95',2:'150'}[i]
                    band2 = {0:'95',1:'150',2:'150'}[i]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    # print('dltt 0-10',dltt[0:10])
                    if comp == 'primary':
                        dls[i] = (dltt ) * c1 * c2
                    else:
                        dls[i] = (self.get_power('TT',[comp],params,
                                                eff_freq_ghz1=self.effs[band1],array1=None,
                                                eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2
                    # print('dls 0-10',dls[0:10])
                elif i>=3 and i<=6:
                    band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                    band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y = params[f'yp_{band2}']
                    if comp == 'primary':
                        dls[i] = (dlte  ) * c1 * c2 * y
                    else:
                        dls[i] = (self.get_power('TE',[comp],params,
                                                    eff_freq_ghz1=self.effs[band1],array1=None,
                                                    eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2 * y
                else:
                    band1 = {0:'95',1:'95',2:'150'}[i-7]
                    band2 = {0:'95',1:'150',2:'150'}[i-7]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y1 = params[f'yp_{band1}']
                    y2 = params[f'yp_{band2}']
                    if comp == 'primary':
                        dls[i] =  (dlee ) * c1 * c2 * y1 * y2
                    else:
                        dls[i] =  (self.get_power('EE',[comp],params,
                                                    eff_freq_ghz1=self.effs[band1],array1=None,
                                                    eff_freq_ghz2=self.effs[band2],array2=None,lmax=lkl_setup.sp.l_max,ptsz=ptsz) ) * c1 * c2 * y1 * y2
                # print('dls 0-10',dls[i][0:10])
            return bin_func(dls/ells/(ells+1.)*2.*np.pi)




    def get_coadd_power(self,cdata,ibbl,ells,dl,spec,fparams,lkl_setup = None,ptsz=None):

        icov,icov_ibin,Pmat,arrays = cdata

        l_max = lkl_setup.sp.l_max

        ps = []
        for row in arrays:
            ind,r,season1,season2,array1,array2 = row
            a1 = '_'.join([season1,array1])
            a2 = '_'.join([season2,array2])

            pow = dl + self.get_power(spec,self.comps,fparams,
                                      eff_freq_ghz1=None,array1=a1,
                                      eff_freq_ghz2=None,array2=a2,
                                      lmax=l_max,
                                      ptsz=ptsz) # 7924 for act alone, 3924 for act_planck !ell max of the full window functions

            pow = pow/ells/(ells+1)*2.*np.pi
            bpow = np.einsum('...k,...k',ibbl,pow)
            ps = np.append(ps,bpow.copy())
        # exit(0)


        return np.dot(icov_ibin,np.dot(Pmat,np.dot(icov,ps)))


    def get_coadd_power_comp(self,cdata,ibbl,ells,dl,spec,fparams,lkl_setup = None,comp = None,ptsz=None):

        icov,icov_ibin,Pmat,arrays = cdata

        l_max = lkl_setup.sp.l_max

        ps = []
        for row in arrays:
            ind,r,season1,season2,array1,array2 = row
            a1 = '_'.join([season1,array1])
            a2 = '_'.join([season2,array2])
            # print('a1,a2:',a1,a2)
            if comp == 'primary':
                pow = dl
            else:
                pow = self.get_power(spec,[comp],fparams,
                                          eff_freq_ghz1=None,array1=a1,
                                          eff_freq_ghz2=None,array2=a2,
                                          lmax=l_max,
                                          ptsz=ptsz) # 7924 for act alone, 3924 for act_planck

            pow = pow/ells/(ells+1)*2.*np.pi
            bpow = np.einsum('...k,...k',ibbl,pow)
            ps = np.append(ps,bpow.copy())


        return np.dot(icov_ibin,np.dot(Pmat,np.dot(icov,ps)))




    def get_theory_bandpassed_parallel(self,i,coadd_data,ells,bbl,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None):
        dim = lkl_setup.sp.n_bins*lkl_setup.sp.n_specs
        sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
        assert len(self.cache['CIB'])==0

        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0
        cls = np.zeros((dim,))
        if lkl_setup.use_act_planck == 'yes':


            spec = 'TT'
            band1 = lkl_setup.sp.fband1[i]
            band2 = lkl_setup.sp.fband2[i]
            c1 = params[f'cal_{band1}']
            c2 = params[f'cal_{band2}']

            cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup) * c1 * c2
            # exit(0)
        else:

            if i<3:
                spec = 'TT'
                band1 = {0:'95',1:'95',2:'150'}[i]
                band2 = {0:'95',1:'150',2:'150'}[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']

                cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup) * c1 * c2

            elif i>=3 and i<=6:
                spec = 'TE'
                band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']
                y = params[f'yp_{band2}']

                cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlte,spec,params,lkl_setup) * c1 * c2 * y

            else:
                spec = 'EE'
                band1 = {0:'95',1:'95',2:'150'}[i-7]
                band2 = {0:'95',1:'150',2:'150'}[i-7]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']
                y1 = params[f'yp_{band1}']
                y2 = params[f'yp_{band2}']

                cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlee,spec,params,lkl_setup) * c1 * c2 * y1 * y2



        self.cache['CIB'] = {}
        return cls



    def get_theory_bandpassed(self,coadd_data,ells,bbl,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None,ptsz = None):
        dim = lkl_setup.sp.n_bins*lkl_setup.sp.n_specs
        assert len(self.cache['CIB'])==0

        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0
        cls = np.zeros((dim,))
        if lkl_setup.use_act_planck == 'yes':

            for i in range(lkl_setup.sp.n_specs):
                sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
                # print(sel)


                spec = 'TT'
                band1 = lkl_setup.sp.fband1[i]
                band2 = lkl_setup.sp.fband2[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']

                cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup,ptsz) * c1 * c2
            # exit(0)
        else:

            for i in range(lkl_setup.sp.n_specs):
                sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]

                if i<3:
                    spec = 'TT'
                    band1 = {0:'95',1:'95',2:'150'}[i]
                    band2 = {0:'95',1:'150',2:'150'}[i]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']

                    cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup,ptsz) * c1 * c2

                elif i>=3 and i<=6:
                    spec = 'TE'
                    band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                    band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y = params[f'yp_{band2}']

                    cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlte,spec,params,lkl_setup,ptsz) * c1 * c2 * y

                else:
                    spec = 'EE'
                    band1 = {0:'95',1:'95',2:'150'}[i-7]
                    band2 = {0:'95',1:'150',2:'150'}[i-7]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y1 = params[f'yp_{band1}']
                    y2 = params[f'yp_{band2}']

                    cls[sel] = self.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlee,spec,params,lkl_setup,ptsz) * c1 * c2 * y1 * y2



        self.cache['CIB'] = {}
        return cls

    def get_theory_bandpassed_comp(self,coadd_data,ells,bbl,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None,comp=None,ptsz=None):

        assert len(self.cache['CIB'])==0

        if lmax is not None:
            dltt[ells>lmax] = 0
            dlte[ells>lmax] = 0
            dlee[ells>lmax] = 0

        if lkl_setup.use_act_planck == 'yes':
            # print(lkl_setup.sp.n_bins)
            # print(lkl_setup.sp.n_specs)
            dim = lkl_setup.sp.n_bins*lkl_setup.sp.n_specs
            # exit(0)
            cls = np.zeros((dim,))
            for i in range(lkl_setup.sp.n_specs):
                sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
                # print(sel)


                spec = 'TT'
                band1 = lkl_setup.sp.fband1[i]
                band2 = lkl_setup.sp.fband2[i]
                c1 = params[f'cal_{band1}']
                c2 = params[f'cal_{band2}']

                cls[sel] = self.get_coadd_power_comp(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup,comp,ptsz) * c1 * c2
            # exit(0)
        else:
            cls = np.zeros((520,))
            for i in range(10):
                sel = np.s_[i*52:(i+1)*52]

                if i<3:
                    spec = 'TT'
                    band1 = {0:'95',1:'95',2:'150'}[i]
                    band2 = {0:'95',1:'150',2:'150'}[i]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']

                    cls[sel] = self.get_coadd_power_comp(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup,comp,ptsz) * c1 * c2

                elif i>=3 and i<=6:
                    spec = 'TE'
                    band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
                    band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y = params[f'yp_{band2}']

                    cls[sel] = self.get_coadd_power_comp(coadd_data[spec][(band1,band2)],bbl[i],ells,dlte,spec,params,lkl_setup,comp,ptsz) * c1 * c2 * y

                else:
                    spec = 'EE'
                    band1 = {0:'95',1:'95',2:'150'}[i-7]
                    band2 = {0:'95',1:'150',2:'150'}[i-7]
                    c1 = params[f'cal_{band1}']
                    c2 = params[f'cal_{band2}']
                    y1 = params[f'yp_{band1}']
                    y2 = params[f'yp_{band2}']

                    cls[sel] = self.get_coadd_power_comp(coadd_data[spec][(band1,band2)],bbl[i],ells,dlee,spec,params,lkl_setup,comp,ptsz) * c1 * c2 * y1 * y2



        self.cache['CIB'] = {}
        return cls
