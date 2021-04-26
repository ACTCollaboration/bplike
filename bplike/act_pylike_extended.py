import numpy as np
#from cobaya.conventions import _path_install
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.likelihoods._base_classes import _InstallableLikelihood
import os,sys
from .config import *
from .utils import *
from .fg import *
# import utils
# import fg
from soapack import interfaces as sints
from pkg_resources import resource_filename

save_coadd_data = False
save_coadd_data_extended = False
# run with:
# $ cobaya-run run_scripts/act_extended_act_plus_planck.yml -f


dfroot = resource_filename("bplike","data/actpolfull_dr4.01/data/data_act/")

# dfroot_coadd_w = resource_filename("bplike","data")+"/bplike_data/big_coadd_weights/200226/"
# dfroot_coadd_d = resource_filename("bplike","data")+"/bplike_data/coadd_data/"
# dfroot_fg = resource_filename("bplike","data")+"/actpolfull_dr4.01/data/Fg/"
# dfroot_bpass = resource_filename("bplike","data")+"/bplike_data/bpass/"


dfroot = path_to_data+"/actpolfull_dr4.01/data/data_act/"

dfroot_coadd_w = path_to_data+"/bplike_data/big_coadd_weights/200226/"
dfroot_coadd_d = path_to_data+"/bplike_data/coadd_data/"
dfroot_fg = path_to_data+"/actpolfull_dr4.01/data/Fg/"
dfroot_bpass = path_to_data+"/bplike_data/bpass/"

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

def get_band_extended(array):
    a = array.split("_")[1]
    assert a[0]=='f'
    if a[1:]=='150':
        return '150'
    elif a[1:]=='090':
        return '090'
    elif a[1:]=='100':
        return '100'
    elif a[1:]=='143':
        return '143'
    elif a[1:]=='217':
        return '217'
    elif a[1:]=='353':
        return '353'
    elif a[1:]=='545':
        return '545'
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
    # print('regions:',regions)
    for region in regions:
        order = np.load(f"{dfroot_coadd_w}{region}_all_C_ell_data_order_190918.npy")
        df = pd.DataFrame(order,columns=['t1','t2','region','s1','s2','a1','a2']).stack().str.decode('utf-8').unstack()
        df = df[(df.t1==spec[0]) & (df.t2==spec[1]) & (df.region==rmap(region)+"_")]
        # print('region:',region)
        # print('df:',df)
        arrays = []
        for index, row in df.iterrows():
            b1 = get_band(row.a1)
            b2 = get_band(row.a2)
            # print('b1,b2:',b1,b2)
            if (b1==band1 and b2==band2) or (b1==band2 and b2==band1):
                # print('rmap(region):',rmap(region))
                # print('rmap b1,b2:',b1,b2)
                # print('row.s1:',row.s1)
                arrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                barrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))

        adf = pd.DataFrame(arrays,columns = ['i','r','s1','s2','a1','a2'])
        ids = adf.i.to_numpy()
        oids = []
        for ind in ids:
            oids = oids + list(range(ind*nbin+rbin,(ind+1)*nbin))
        # print('oids:',oids)
        # print('soids:',np.shape(oids))

        """
        Covmat selection
        """
        # print('loading cov:',dfroot_coadd_w+f"{region}_all_covmat_190918.npy")
        cov = np.load(dfroot_coadd_w+f"{region}_all_covmat_190918.npy")
        # print('cov loaded')
        # print('scov:',np.shape(cov))
        ocov = cov[oids,:][:,oids]
        # print('socov:',np.shape(ocov))
        icovs.append(np.linalg.inv(ocov))
    # print('sicovs:',np.shape(icovs))
    icov = block_diag(*icovs)
    # print('sicov:',np.shape(icov))
    nspec = 1
    nbins = 52
    norig = len(barrays)
    # print('norig:',norig)
    # print('barrays:',barrays)
    N_spec = nbins*nspec
    # print('N_spec:',N_spec)
    # building projection matrix with shape nbins*nspec x nbins*norig
    pmat = np.identity(N_spec)
    Pmat = np.identity(N_spec)
    for i in range(1,norig):
        Pmat = np.append(Pmat,pmat,axis=1)
    # print('sPmat:',np.shape(Pmat))
    icov_ibin = np.linalg.inv(np.dot(Pmat,np.dot(icov,Pmat.T)))
    # np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov.txt',icov)
    # np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov_ibin.txt',icov_ibin)
    # np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_pmat.txt',Pmat)
    barrays = np.array(barrays,dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')])
    # np.savetxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_arrays.txt', barrays,fmt=['%d','%s','%s','%s','%s','%s'])


def load_coadd_matrix(spec,band1,band2,flux,path_root):
    icov = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov.txt',)
    icov_ibin = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_icov_ibin.txt',)
    pmat = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_pmat.txt',)
    arrays = np.loadtxt(f'{path_root}_{spec}_{band1}_{band2}_{flux}_arrays.txt',dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')],ndmin=1)
    return icov,icov_ibin,pmat,arrays


class StevePower(object):
    def __init__(self,froot,flux,infval=1e10,tt_lmin=600,tt_lmax=None):
        # print('doing stevepower')
        spec=np.loadtxt(f"{froot}coadd_cl_{flux}_data_200124.txt")
        cov =np.loadtxt(f'{froot}coadd_cov_{flux}_200519.txt')
        bbl = np.loadtxt(f'{froot}coadd_bpwf_{flux}_191127_lmin2.txt')
        # print('bbl shape:',np.shape(bbl))
        # exit(0)
        self.bbl =bbl.reshape((10,52,7924))
        self.spec = spec[:520]
        self.cov = cov[:520,:520]
        nbin = 52
        self.n_bins = nbin
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
            # print('cov')
            # print(cov)

        if tt_lmax is not None:
            n = 3
            ids = []
            ids = np.argwhere(self.ls>tt_lmax)[:,0]
            ids = ids[ids<nbin*3]
            self.cov[:,ids] = 0
            self.cov[ids,:] = 0
            self.cov[ids,ids] = infval
            # print('cov')
            # print(cov)

        self.cinv = np.linalg.inv(self.cov)

        if save_coadd_data == True:
            band1 = '95'
            band2 = '95'
            flux = '100mJy'
            path_root = None
            spec = 'TT'

            save_coadd_matrix(spec,band1,band2,flux,path_root)

    def bin(self,dls):
        # print('bbl in bin, size : ',len(self.bbl[0,0,:]) )
        # print(self.bbl[0,0,:])
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


    # exit()



class StevePower_extended(object):
    def __init__(self,data_root,flux,infval=1e10,tt_lmin=600,tt_lmax=None):
        # data_root = path_to_data + '/act_planck_data_210328/'
        specs = ['f090xf090','f090xf100','f090xf143','f090xf150',
         'f090xf217','f090xf353','f090xf545','f100xf100',
         'f100xf143','f143xf143','f100xf150','f143xf150',
         'f150xf150','f150xf217','f150xf353','f150xf545',
         'f100xf217','f143xf217','f217xf217','f100xf353',
         'f143xf353','f217xf353','f353xf353','f100xf545',
         'f143xf545','f217xf545','f353xf545','f545xf545']
        specx_l = ['TT']

        freqs_asked = []
        fband1 = []
        fband2 = []
        for spec in specs:
            comp1 = spec.split('x')[0]
            comp2 = spec.split('x')[1]
            comp1 = comp1.replace('f', '')
            comp2 = comp2.replace('f', '')
            fband1.append(comp1)
            fband2.append(comp2)
            # print(comp1,comp2)
            if comp1 not in freqs_asked:
                freqs_asked.append(comp1)
            if comp2 not in freqs_asked:
                freqs_asked.append(comp2)
        freqs_asked.sort()
        # print('fband1: ')
        # print(fband1)
        # print('fband2: ')
        # print(fband2)
        self.fband1 = fband1
        self.fband2 = fband2
        # print('frequency list used in the analysis: ')
        # print(freqs_asked)
        self.cfreqs_list = freqs_asked

        if flux == '15mJy':
            rfroot = 'deep56'
        if flux == '100mJy':
            rfroot = 'boss'

        spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210327.npy')
        cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210327.npy')
        covx = np.load(data_root+f'{rfroot}_all_covmat_anal_210327.npy')
        bbl = np.load(data_root+f'{rfroot}_bpwf_210327.npy')

        l_min = 2
        # print('bbl shape:',np.shape(bbl))
        n_specs = len(specs)
        # print('n_specs: ',n_specs)
        n_bins = int(len(spec)/n_specs)
        # print('n_bins: ',n_bins)
        n_ells = np.shape(bbl)[1]
        # print('n_ells: ',n_ells)
        n_ells = n_ells-l_min
        bbl_2 = np.zeros((n_bins*n_specs,n_ells))
        for i in range(n_bins*n_specs):
            bbl_2[i,:] = np.delete(bbl[i,:],[0,1])
        # print(len(bbl_2[0,:]))


        self.n_specs = n_specs


        self.n_bins = n_bins

        # doing coadd_data
        #if self.bandpass:
        if save_coadd_data_extended:
            from scipy.linalg import block_diag
            import pandas as pd
            if flux=='15mJy':
                regions = ['deep56']
            elif flux=='100mJy':
                regions = ['boss'] + [f'advact_window{x}' for x in range(6)]
            else:
                raise ValueError
            # order = spec
            # print('starting df for region:',regions)
            region = regions[0]
            specx = specx_l[0]
            order = np.load(data_root+f'{rfroot}_all_C_ell_data_order_190918.npy')
            df = pd.DataFrame(order,columns=['t1','t2','region','s1','s2','a1','a2'])#.stack().str.decode('utf-8').unstack()
            # print(df)
            def rmap(r):
                if r[:6]=='advact': return 'advact'
                else: return r
            # print('restricting df')
            df = df[(df.t1==specx[0]) & (df.t2==specx[1]) & (df.region==rmap(region)+"_")]
            # df = df[(df.t1==spec[0]) & (df.t2==spec[1]) & (df.region==rmap(region)+"_")]
            # print(df)
            # print('  ')
            # print('  ')
            for ib in range(28):
                # print('  ')
                # print('  ')

                band1 = fband1[ib]
                band2 = fband2[ib]

                arrays = []
                barrays = []
                icovs = []
                nbin = self.n_bins+3
                # print('ib b1, b2, nbin:',ib,band1,band2,nbin)
                rbin = 3
                # normallly here there is loop over region
                ibx = 0
                for index, row in df.iterrows():
                    # print('id,row:',index,row)
                    b1 = get_band_extended(row.a1)
                    b2 = get_band_extended(row.a2)
                    # print('b1,b2:',b1,b2)

                    if (b1==band1 and b2==band2) or (b1==band2 and b2==band1):
                        #print('row:',row)
                        # print('rmap(region):',index,ibx,rmap(region))
                        # print('rmap b1,b2:',b1,b2)
                        # print('row.s1:',row.s1)
                        # print('row.s2:',row.s2)
                        # print('row.a1:',row.a1)
                        # print('row.a2:',row.a2)
                        # print(' ')
                        # print('#####')
                        arrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                        barrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                        ibx += 1
                adf = pd.DataFrame(arrays,columns = ['i','r','s1','s2','a1','a2'])
                ids = adf.i.to_numpy()
                oids = []
                for ind in ids:
                    oids = oids + list(range(ind*nbin+rbin,(ind+1)*nbin))
                # print('oids:',oids)
                # print('scov:',np.shape(cov))
                # print('soids:',np.shape(oids))
                # try:
                ocov = covx[oids,:][:,oids]

                #exit(0)
                # print('socov:',np.shape(ocov))
                icovs.append(np.linalg.inv(ocov))
                # print('sicovs:',np.shape(icovs))
                icov = block_diag(*icovs)
                # print('sicov:',np.shape(icov))
                # print('barrays:',barrays)
                nspec = 1
                nbins = self.n_bins
                norig = len(barrays)
                # print('norig:',norig)
                # print('barrays:',barrays)
                N_spec = nbins*nspec
                # print('N_spec:',N_spec)
                pmat = np.identity(N_spec)
                Pmat = np.identity(N_spec)
                for i in range(1,norig):
                    Pmat = np.append(Pmat,pmat,axis=1)
                # print('sPmat:',np.shape(Pmat))
                icov_ibin = np.linalg.inv(np.dot(Pmat,np.dot(icov,Pmat.T)))
                barrays = np.array(barrays,dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')])
                # print('barrays:',barrays)
                np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_icov.txt',icov)
                np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_icov_ibin.txt',icov_ibin)
                np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_pmat.txt',Pmat)
                np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_arrays.txt', barrays,fmt=['%d','%s','%s','%s','%s','%s'])

                # print('ib2:',ib,band1,band2)
            #
        #exit(0)






        self.bbl = bbl_2.reshape((n_specs,n_bins,n_ells))
        self.spec = spec[:,1]
        self.cov = cov
        # print('shape cov : ', np.shape(self.cov))
        nbin = n_bins
        #self.ells = np.arange(2,n_ells+2)
        self.ells = np.arange(l_min,n_ells+2)
        # self.ells = np.arange(2,7924+2)
        # self.ells = spec[:,0]
        #rells = np.repeat(self.ells[None],n_specs,axis=0)
        #print(np.shape(rells))
        #self.ls = self.bin(rells)
        self.ls = spec[:,0]

        # conversion factor to bplike normalisations, i.e., dl's to cl's:
        fac = self.ls*(self.ls+1.)/2./np.pi
        self.spec = self.spec/fac
        # print('shape spec : ', np.shape(self.spec))
        # print('shape fac : ', np.shape(fac))
        self.cov = self.cov/fac**2.



        if tt_lmin is not None:
            # n = 3
            ids = []
            ids = np.argwhere(self.ls<tt_lmin)[:,0]
            # print('ls: ', self.ls)
            # print(len(self.ls))
            # print('nbins: ',nbin)
            # print('ids ls<tt_lmin: ', ids)
            # print('fband1 : ',self.fband1)
            rfband1 = np.repeat(self.fband1,nbin)
            rfband2 = np.repeat(self.fband2,nbin)
            # print(len(rfband2))
            # print(type(rfband2[0]))
            # cd_act =
            ids_act = np.argwhere((self.ls<tt_lmin) & ((rfband1 == '090') | (rfband1 == '150') | (rfband2 == '090') | (rfband2 == '150')))[:,0]
            # print('idsact : ',ids_act)
            # exit(0)
            ids = ids_act
            self.cov[:,ids] = 0
            self.cov[ids,:] = 0
            self.cov[ids,ids] = infval
            # print('cov')
            # print(cov)

        # if tt_lmax is not None:
        #     n = 3
        #     ids = []
        #     ids = np.argwhere(self.ls>tt_lmax)[:,0]
        #     ids = ids[ids<nbin*3]
        #     self.cov[:,ids] = 0
        #     self.cov[ids,:] = 0
        #     self.cov[ids,ids] = infval

        self.cinv = np.linalg.inv(self.cov)

    def bin(self,dls):
        # print('bbl in bin, size : ',len(self.bbl[0,0,:]) )
        # print(self.bbl[0,0,:])
        bdl = np.einsum('...k,...k',self.bbl,dls[:,None,:])
        return bdl.reshape(-1)

    #
    # def select(self,bls,spec,band1,band2,shift=52):
    #     I = {'tt':0,'te':3,'ee':7}
    #     i = { 'tt':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2},
    #           'te':{('95','95'): 0,('95','150'): 1,('150','95'): 2,('150','150'): 3},
    #           'ee':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2} }
    #     mind = i[spec][(band1,band2)]
    #     sel = np.s_[(I[spec]+mind)*shift:(I[spec]+mind+1)*shift]
    #     if bls.ndim==1: return bls[sel]
    #     elif bls.ndim==2: return bls[sel,sel]
    #     else: raise ValueError



class act_pylike_extended(_InstallableLikelihood):


    def initialize(self):
        # print('initializing')
        # exit(0)

        # self.l_max = 3826
        # self.l_max = 6000

        self.log.info("Initialising.")
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
            "a_s_ee"] # EE Synchrotron at ell=500
        # "cal_95",
        # "cal_150",
        # "yp_95",
        # "yp_150"]
        self.cal_yp_act_only =[
            "cal_95",
            "cal_150",
            "yp_95",
            "yp_150"]
        self.cal_yp_act_plus_planck =[
              "cal_090",
              "yp_090",
              "cal_100",
              "yp_100",
              "cal_143",
              "yp_143",
              "cal_150",
              "yp_150",
              "cal_217",
              "yp_217",
              "cal_353",
              "yp_353",
              "cal_545",
              "yp_545"
              ]
        file = resource_filename("bplike","act_pylike_extended_full.yaml")
        with open(file) as f:
            act_pylike_extended_full = yaml.load(f, Loader=yaml.FullLoader)
        # print('act_pylike_extended_full')
        # print(act_pylike_extended_full)
        # exit(0)
        # Load path_params from yaml file
        if self.use_act_planck == 'no':
            self.l_max = 6000
            self.fparams = config_from_yaml('params.yml')['fixed']
            self.aparams = config_from_yaml('params.yml')['act_like']
            self.bpmodes = config_from_yaml('params.yml')['bpass_modes']
            # for the act only lkl:

            cal_yp =  self.cal_yp_act_only
            # for s in self.cal_yp_act_plus_planck:
            #     if s not in self.cal_yp_act_only:
            #         act_pylike_extended_full['params'].pop(s,None)
            # print('act_pylike_extended_full act only')
            # print(act_pylike_extended_full)

        elif self.use_act_planck == 'yes':
            self.l_max = 3899
            self.fparams = config_from_yaml('params_extended.yml')['fixed']
            self.aparams = config_from_yaml('params_extended.yml')['act_like']
            self.bpmodes = config_from_yaml('params_extended.yml')['bpass_modes']
            # for the act+planck lkl:

            cal_yp = self.cal_yp_act_plus_planck
            # for s in self.cal_yp_act_only:
            #     if s not in self.cal_yp_act_plus_planck:
            #         act_pylike_extended_full['params'].pop(s,None)
            # print('act_pylike_extended_full act+planck')
            # print(act_pylike_extended_full)
        new_file = file.replace('_full', '')
        # print('new_file')
        # print(new_file)
        with open(new_file, 'w') as f:
            yaml.dump(act_pylike_extended_full, f)
        # exit(0)




        self.expected_params = list(np.concatenate((self.expected_params,cal_yp)))
        # print('expected params: ',self.expected_params)
        self.bands = self.aparams['bands']


        # Read data
        # print('preparing data')
        self.prepare_data()

        # State requisites to the theory code
        self.requested_cls = ["tt", "te", "ee"]






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


            # self.input_params.remove('cal_95','yp_95')

        # print('expected params: ',self.expected_params)

        differences = are_different_params_lists(
            self.input_params, self.expected_params,
            name_A="given", name_B="expected")
        if differences:
            # self.input_params = self.expected_params
            raise LoggedError(
                self.log, "Configuration error in parameters: %r.",
                differences)

    def get_requirements(self):
        return {'Cl': {'tt': self.l_max,'te': self.l_max,'ee': self.l_max}}

    def logp(self, **params_values):
        # return 0
        # print('doing logp')
        cl = self.theory.get_Cl(ell_factor=True)
        # print("cl's: ",cl)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        # print('doing loglike')
        # print('cls 0:10:', cl['tt'][:10] )


        ps = self._get_power_spectra(cl, lkl_setup = self, **params_values)
        ps_vec = ps['tot']
        ps_vec_galdust = ps['galdust']
        ps_vec_primary = ps['primary']
        # print('shape ls: ',np.shape(self.sp.ls))
        # print('shape ps_vec : ', np.shape(ps_vec))
        # print('shape self.sp.spec : ', np.shape(self.sp.spec))
        n_bins = self.sp.n_bins

        if self.use_act_planck == 'yes':
            fac = self.sp.ls*(self.sp.ls+1.)/2./np.pi
            # print('ps_vec : ', ps_vec[:n_bins]/fac[:n_bins])
            # print('self.sp.spec : ', self.sp.spec[:n_bins])
            dls_theory = ps_vec
            np.save(path_to_output+'/dls_theory.npy',dls_theory)
            np.save(path_to_output+'/dls_theory_galdust.npy',ps_vec_galdust)
            np.save(path_to_output+'/dls_theory_primary.npy',ps_vec_primary)
            ls_theory = self.sp.ls
            np.save(path_to_output+'/ls_theory.npy',ls_theory)
            delta = self.sp.spec - dls_theory/fac


        elif self.use_act_planck == 'no':
            # print('ps_vec : ', ps_vec[:n_bins])
            # print('self.sp.spec : ', self.sp.spec[:n_bins])
            delta = self.sp.spec - ps_vec
        logp = -0.5 * np.dot(delta,np.dot(self.sp.cinv,delta))
        self.log.debug(
            f"ACT-like {self.flux} lnLike value = {logp} (chisquare = {-2 * logp})")
        return logp

    def prepare_data(self, verbose=False):
        str_current = '[bplike prepare_data] '
        flux = self.flux
        # exit(0)
        # self.sp = StevePower("data/actpol_2f_full_s1316_2flux_fin/data/data_act/ps_200519/",self.flux)
        if self.use_act_planck == 'no':
            data_root = dfroot
            print(str_current+'Collecting power spectra from %s and with flux %s'%(data_root,self.flux))
            self.sp = StevePower(data_root,self.flux)
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
                    # print('freq:',freq)
                    # print('bp files:',bp_dict[pname])
                    # print('beam files:',beam_dict[pname])
                # print('cfreq_dict:',cfreq_dict )
                # print('beam_dict:',beam_dict )
                # print('bp/beam files loaded')
                # print('###################')
                # print('###################')
                # print('###################')
                # exit(0)
                #exit(0)
            else:
                print(str_current+'Not using bandpass - set in the param file if you want to include these.')
                pnames = None
                bp_dict = None
                beam_dict = None
                cfreq_dict = None
        else:
            data_root = path_to_data + '/act_planck_data_210328/'
            # print(str_current+'Collecting power spectra from %s and with flux %s'%(data_root,self.flux))
            self.sp = StevePower_extended(data_root,self.flux)
        # exit(0)
            if self.bandpass:
                # print(' ')
                # print(' ')
                # print(' ')
                # print('doing bandpasses')
                # sbands = { 'TT':[('95','95'),('95','150'),('150','150')],
                #            'TE':[('95','95'),('95','150'),('150','95'),('150','150')],
                #            'EE':[('95','95'),('95','150'),('150','150')] }
                self.coadd_data = {}
                for spec in ['TT']:
                    self.coadd_data[spec] = {}
                    # for bands in sbands[spec]:
                    # print('looping over %d spectra'%self.sp.n_specs)
                    for i in range(self.sp.n_specs):
                        # band1,band2 = bands
                        band1 = self.sp.fband1[i]
                        band2 = self.sp.fband2[i]
                        # print('spec:',spec)

                        # print('dfroot_coadd_d:',dfroot_coadd_d)
                        # print()
                        bands = (band1,band2)
                        # print('bands:',band1,band2)
                        data_root = path_to_data + '/act_planck_data_210328/'
                        if flux=='15mJy':
                            reg = 'deep56'
                        elif flux=='100mJy':
                            reg = 'boss'
                        self.coadd_data[spec][bands] = load_coadd_matrix(spec,band1,band2,
                                                                  self.flux,f"{data_root}{reg}")
                # exit(0)

                # print('coadds matrix loaded')
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
                # print(anames)
                # exit(0)

                pnames = []
                for aname in anames:
                    # print('aname:',aname)

                    season,array,freq,patch = sints.arrays(aname,'season'),sints.arrays(aname,'array'),sints.arrays(aname,'freq'),sints.arrays(aname,'region')
                    # print('season:',season)
                    # print('patch:',patch)
                    # print('freq:',freq)
                    pname = '_'.join([season,array,freq])
                    pnames.append(pname)
                    # print('pname:',pnames)
                    beam_dict[pname] = dm.get_beam_fname(season,patch,array+"_"+freq, version=None)
                    # print('beam_dict[pname]:',beam_dict[pname])

                    bp_dict[pname] = dfroot_bpass+dm.get_bandpass_file_name(array+"_"+freq)
                    # print('bp_dict[pname]:',bp_dict[pname])
                    #col0: freq in GHz
                    #col1: 90 GHz response
                    #col2: 1-sigma error
                    cfreq_dict[pname] = cfreqs[array + "_" + freq]
                # print('  ')
                # print('  ')
                # print('doing planck part:')
                data_root = path_to_data + '/act_planck_data_210328/'
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

                beam_dict['s12_pa0_f100'] = data_root + 'HFI_BEAM_F100.txt'
                beam_dict['s12_pa0_f143'] = data_root + 'HFI_BEAM_F143.txt'
                beam_dict['s12_pa0_f217'] = data_root + 'HFI_BEAM_F217.txt'
                beam_dict['s12_pa0_f353'] = data_root + 'HFI_BEAM_F353.txt'
                beam_dict['s12_pa0_f545'] = data_root + 'HFI_BEAM_F545.txt'

                cfreq_dict['s12_pa0_f100'] =cfreqs['pa0_f100']
                cfreq_dict['s12_pa0_f143'] =cfreqs['pa0_f143']
                cfreq_dict['s12_pa0_f217'] =cfreqs['pa0_f217']
                cfreq_dict['s12_pa0_f353'] =cfreqs['pa0_f353']
                cfreq_dict['s12_pa0_f545'] =cfreqs['pa0_f545']
                # print('  ')
                # print('  ')
                # print('beam_dict:',beam_dict)
                # print('bp_dict:',bp_dict)
                # print('cfreq_dict:',cfreq_dict)

                # exit(0)
            else:
                print(str_current+'Not using bandpass - set in the param file if you want to include these.')
                pnames = None
                bp_dict = None
                beam_dict = None
                cfreq_dict = None

        # print(str_current+'frequencies: ')
        # print('getting foreground power with cfrq: ', cfreq_dict)
        # print('bp_dict: ', bp_dict)
        # print('beam_dict: ', beam_dict)
        # print('flux:',self.flux)
        # print('pnammes:',pnames)
        # exit(0)

        self.fgpower = ForegroundPowers(self.fparams,self.sp.ells,
                                            sz_temp_file,
                                            ksz_temp_file,
                                            sz_x_cib_temp_file,
                                            flux_cut=self.flux,
                                            arrays=pnames,
                                            bp_file_dict=bp_dict,
                                            beam_file_dict=beam_dict,
                                            cfreq_dict=cfreq_dict,
                                            lkl_setup = self)

        # print('foreground power loaded')
        # print('###################')
        # print('###################')
        # print('###################')
        # exit(0)

    def _get_power_spectra(self, cl, lkl_setup = None, **params_values):
        # print('getting power spectra')
        l_min = 2

        if self.theory_debug is not None:
            # print('theory debug')
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
        # print('nells_camb: ', nells_camb)
        nells = self.sp.ells.size
        # print('self.sp.ells.size: ', nells)
        # exit(0)
        # print('camb l0,l1: ',cl['ell'][0],cl['ell'][1])
        # print('camb l-1,l-2: ',cl['ell'][-1],cl['ell'][-2])
        # print('sp.ells l0,l1: ',self.sp.ells[0],self.sp.ells[1])
        # print('sp.ells l-1,l-2: ',self.sp.ells[-1],self.sp.ells[-2])
        assert cl['ell'][0]==0
        assert cl['ell'][1]==1
        assert self.sp.ells[0]==l_min
        assert self.sp.ells[1]==l_min + 1
        ptt = np.zeros(nells+l_min)
        pte = np.zeros(nells+l_min)
        pee = np.zeros(nells+l_min)
        ptt[l_min:nells_camb] = cl['tt'][l_min:]
        pte[l_min:nells_camb] = cl['te'][l_min:]
        pee[l_min:nells_camb] = cl['ee'][l_min:]

        if self.bandpass:
            # print('doing theory bandpass')
            fpower = self.fgpower.get_theory_bandpassed(self.coadd_data,
                                                        self.sp.ells,
                                                        self.sp.bbl,
                                                        ptt[l_min:],
                                                        pte[l_min:],
                                                        pee[l_min:],
                                                        fgdict,
                                                        lmax=self.l_max,
                                                        lkl_setup = lkl_setup)
            return {'tot': fpower,
                    'primary': fpower,
                    'galdust': fpower}


        else:
            # print('doing theory no bandpass')
            # print('sp.ells: ',self.sp.ells)
            fpower = self.fgpower.get_theory(self.sp.ells,
                                             self.sp.bin,
                                             ptt[l_min:],
                                             pte[l_min:],
                                             pee[l_min:],
                                             fgdict,
                                             lmax=self.l_max,
                                             lkl_setup = lkl_setup)


            fpower_primary = self.fgpower.get_primary(self.sp.ells,
                                             self.sp.bin,
                                             ptt[l_min:],
                                             pte[l_min:],
                                             pee[l_min:],
                                             fgdict,
                                             lmax=self.l_max,
                                             lkl_setup = lkl_setup)

            fpower_galdust = self.fgpower.get_galdust(self.sp.ells,
                                             self.sp.bin,
                                             ptt[l_min:],
                                             pte[l_min:],
                                             pee[l_min:],
                                             fgdict,
                                             lmax=self.l_max,
                                             lkl_setup = lkl_setup)
            # print('fpower : ', fpower[0:10])
            # exit(0)

            return {'tot': fpower,
                    'primary': fpower_primary,
                    'galdust': fpower_galdust}



class act15(act_pylike_extended):
    flux = '15mJy'
    use_act_planck = 'None'

class act100(act_pylike_extended):
    flux = '100mJy'
    use_act_planck = 'None'
