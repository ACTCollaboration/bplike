import numpy as np
#from cobaya.conventions import _path_install
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists
from cobaya.likelihoods.base_classes import InstallableLikelihood
import os,sys
from .config import *
from .utils import *
from .fg import *
# import utils
# import fg
from soapack import interfaces as sints
from pkg_resources import resource_filename

import multiprocessing
import functools

infval = 1e300

class StevePower(object):
    def __init__(self,froot,flux,infval=infval,tt_lmin=600,tt_lmax=None,l_max_data = 0):
        # print('doing stevepower')
        spec=np.loadtxt(f"{froot}coadd_cl_{flux}_data_200124.txt")
        cov =np.loadtxt(f'{froot}coadd_cov_{flux}_200519.txt')
        bbl = np.loadtxt(f'{froot}coadd_bpwf_{flux}_191127_lmin2.txt')
        self.l_max = l_max_data

        if flux=='15mJy':
            lregion = 'deep'
        elif flux=='100mJy':
            lregion = 'wide'
        self.leak_95,self.leak_150 = np.loadtxt(f'{froot}leak_TE_{lregion}_200519.txt',usecols=[1,2],unpack=True)
        # print('bbl shape:',np.shape(bbl))
        # exit(0)
        self.n_specs = 10
        self.bbl =bbl.reshape((10,52,self.l_max))
        self.bbl[:,:,:100] = 0
        self.spec = spec[:520]
        self.cov = cov[:520,:520]
        nbin = 52
        self.n_bins = nbin
        self.ells = np.arange(2,self.l_max+2)  # ell max of the full window functions = self.l_max
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


        fband1 = []
        fband2 = []
        for i in range(10):
            if i<3:
                fband1.append({0:'T095',1:'T095',2:'T150'}[i])
                fband2.append({0:'T095',1:'T150',2:'T150'}[i])
            elif i>=3 and i<=6:
                fband1.append({0:'T095',1:'T095',2:'T150',3:'T150'}[i-3])
                fband2.append({0:'E095',1:'E150',2:'E095',3:'E150'}[i-3])
            else:
                fband1.append({0:'E095',1:'E095',2:'E150'}[i-7])
                fband2.append({0:'E095',1:'E150',2:'E150'}[i-7])
        self.fband1 = fband1
        self.fband2 = fband2

        self.rfband1 =  np.repeat(self.fband1,nbin)
        self.rfband2 =  np.repeat(self.fband2,nbin)
        # print('rfband1 act:')
        # print(self.rfband1)
        # exit(0)


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
    def __init__(self,data_root,flux,infval=infval,tt_lmin=600,tt_lmax=None,l_max_data = 0, diag_cov_only = False):
        self.l_max = l_max_data
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

        if l_max_data == 7924:
            # spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210610.npy')
            # cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210610.npy')
            spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210702.npy')
            cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210702.npy')
            covx = np.load(data_root+f'{rfroot}_all_covmat_anal_210702.npy')
            bbl = np.load(data_root+f'{rfroot}_bpwf_210610.npy')

            # spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210610.npy')
            # cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210610.npy')
            # covx = np.load(data_root+f'{rfroot}_all_covmat_anal_210610.npy')
            # bbl = np.load(data_root+f'{rfroot}_bpwf_210610.npy')
        else:
            spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210327.npy')
            cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210327.npy')
            covx = np.load(data_root+f'{rfroot}_all_covmat_anal_210327.npy')
            bbl = np.load(data_root+f'{rfroot}_bpwf_210327.npy')

        l_min = 2
        n_specs = len(specs)
        n_bins = int(len(spec)/n_specs)
        n_ells = np.shape(bbl)[1]  # ell max of the full window functions = 3926

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
                        arrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                        barrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
                        ibx += 1
                adf = pd.DataFrame(arrays,columns = ['i','r','s1','s2','a1','a2'])
                ids = adf.i.to_numpy()
                oids = []
                for ind in ids:
                    oids = oids + list(range(ind*nbin+rbin,(ind+1)*nbin))
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
        # cut low part of bpwf
        self.bbl[:,:,:100] = 0


        self.spec = spec[:,1]
        self.cov = cov
        # print('shape cov : ', np.shape(self.cov))
        nbin = n_bins
        #self.ells = np.arange(2,n_ells+2)
        self.ells = np.arange(l_min,n_ells+2)
        # self.ells = np.arange(2,self.l_max+2)
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

        rfband1 = np.repeat(self.fband1,nbin)
        rfband2 = np.repeat(self.fband2,nbin)
        self.rfband1 =  rfband1
        self.rfband2 =  rfband2

        # cut all l<600 for act bands
        if tt_lmin is not None:
            # n = 3
            ids = []
            ids = np.argwhere(self.ls<tt_lmin)[:,0]

            ids_act = np.argwhere((self.ls<tt_lmin) & ((rfband1 == '090') | (rfband1 == '150') | (rfband2 == '090') | (rfband2 == '150')))[:,0]
            ids = ids_act
            self.cov[:,ids] = 0
            self.cov[ids,:] = 0
            self.cov[ids,ids] = infval

        # cutting lower bins (most power is filtered then):
        ids = np.argwhere((self.ls<200))[:,0]
        self.cov[:,ids] = 0
        self.cov[ids,:] = 0
        self.cov[ids,ids] = infval



        # # cutting higher bins for tests:
        # ids = np.argwhere((self.ls>3924))[:,0]
        # self.cov[:,ids] = 0
        # self.cov[ids,:] = 0
        # self.cov[ids,ids] = infval


        # cutting out some spectra
        # list of all available spectra:
        ps_list = ['090x090', '090x100', '090x143', '090x150', '090x217',
        '090x353', '090x545', '100x100', '100x143', '143x143', '100x150', '143x150', '150x150', '150x217',
        '150x353', '150x545', '100x217', '143x217', '217x217', '100x353', '143x353', '217x353',
        '353x353', '100x545', '143x545', '217x545', '353x545', '545x545']
        # here list the spectra that you want to remove,
        ps_list_to_throw = ['090x545','100x545','143x545','150x545','217x545', # thrown because no SNR
                            '353x545','545x545'] # thrown because inconsistent with Choi et al FG modeling
        # ps_list_to_throw = ['']
        for ps in ps_list:
            if ps in ps_list_to_throw:
                ids = np.argwhere( (self.rfband1 == ps.split('x')[0]) & (self.rfband2 == ps.split('x')[1]) )[:,0]
                self.cov[:,ids] = 0
                self.cov[ids,:] = 0
                self.cov[ids,ids] = infval
            else:
                continue


        # # enhancing diagonal to avoid lambda<0:
        # diag_cov_dl = np.diag(np.diagonal(self.cov))
        # self.cov = 1.130*(diag_cov_dl*np.identity(np.shape(self.cov)[0]))+ self.cov - diag_cov_dl

        # keeping only diagonal elements to covmat
        if diag_cov_only:
            self.cov = np.diag(np.diagonal(self.cov))

            # j = label_bps.index(ps)
            # self.cov[j*self.n_bins:(j+1)*self.n_bins,j*self.n_bins:(j+1)*self.n_bins] = infval
            # self.cov[:,j*self.n_bins:(j+1)*self.n_bins] = 0.
            # self.cov[j*self.n_bins:(j+1)*self.n_bins,:] = 0.



        # print('setting inf in cov where beam too small')

        beam_dict_f100 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F100.txt')
        beam_dict_f143 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F143.txt')
        beam_dict_f217 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F217.txt')
        beam_dict_f353 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F353.txt')
        beam_dict_f545 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F545.txt')

        beam_cut_off = 0.1
        lmax_beam_cutoff = {}
        lmax_100 = beam_dict_f100[:,0][beam_dict_f100[:,1]>beam_cut_off].max()
        lmax_143 = beam_dict_f100[:,0][beam_dict_f143[:,1]>beam_cut_off].max()
        lmax_217 = beam_dict_f100[:,0][beam_dict_f217[:,1]>beam_cut_off].max()
        lmax_353 = beam_dict_f100[:,0][beam_dict_f353[:,1]>beam_cut_off].max()
        lmax_545 = beam_dict_f100[:,0][beam_dict_f545[:,1]>beam_cut_off].max()

        lmax_090 = infval
        lmax_150 = infval
        lmax_beam_cutoff['090'] = lmax_090
        lmax_beam_cutoff['150'] = lmax_150
        lmax_beam_cutoff['100'] = lmax_100
        lmax_beam_cutoff['143'] = lmax_143
        lmax_beam_cutoff['217'] = lmax_217
        lmax_beam_cutoff['353'] = lmax_353
        lmax_beam_cutoff['545'] = lmax_545

        lmax_order_list = []

        for (fb1,fb2) in zip(self.fband1,self.fband2):
            lmax_order_list.append(min(lmax_beam_cutoff[fb1],lmax_beam_cutoff[fb2]))


        rlmax_order_list = np.repeat(lmax_order_list,nbin)
        ids_cutoff = np.argwhere((self.ls>rlmax_order_list))[:,0]

        ids = ids_cutoff
        self.cov[:,ids] = 0
        self.cov[ids,:] = 0
        self.cov[ids,ids] = infval



        self.cinv = np.linalg.inv(self.cov)

    def bin(self,dls):
        # print('bbl in bin, size : ',len(self.bbl[0,0,:]) )
        # print(self.bbl[0,0,:])
        bdl = np.einsum('...k,...k',self.bbl,dls[:,None,:])
        return bdl.reshape(-1)
