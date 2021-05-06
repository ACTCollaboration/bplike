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


#
#
#
# def get_theory_bandpassed_parallel_2(i,coadd_data):
#     return 0
#
# def get_theory_bandpassed_parallel(i,fgpower,coadd_data,ells,bbl,dltt,dlte,dlee,params,lmax=6000,lkl_setup = None):
#     dim = lkl_setup.sp.n_bins*lkl_setup.sp.n_specs
#     sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
#     assert len(fgpower.cache['CIB'])==0
#
#     if lmax is not None:
#         dltt[ells>lmax] = 0
#         dlte[ells>lmax] = 0
#         dlee[ells>lmax] = 0
#     cls = np.zeros((dim,))
#     if lkl_setup.use_act_planck == 'yes':
#         # print(lkl_setup.sp.n_bins)
#         # print(lkl_setup.sp.n_specs)
#
#         # exit(0)
#
#         # for i in range(lkl_setup.sp.n_specs):
#         # sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
#         # print(sel)
#
#
#         spec = 'TT'
#         band1 = lkl_setup.sp.fband1[i]
#         band2 = lkl_setup.sp.fband2[i]
#         c1 = params[f'cal_{band1}']
#         c2 = params[f'cal_{band2}']
#         # print('  ')
#         # print('  ')
#         # print('getting theory bp at:')
#         # print(c1,c2,band1,band2)
#         cls[sel] = fgpower.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup) * c1 * c2
#         # exit(0)
#     else:
#         # cls = np.zeros((520,))
#
#         # parallel compoutation:
#
#
#         # a_pool = multiprocessing.Pool()
#         # print('starting pool')
#         #
#         # p1 = 0
#         # pool = multiprocessing.Pool()
#         # fn = functools.partial(get_coadd_power_act_only_parallel,p1=p1)
#         # print('pool.map')
#         # r = pool.map(fn,range(10))
#         #                             # self=self,
#         #                             # cls=cls,
#         #                             # coadd_data=coadd_data,
#         #                             # bbl=bbl,
#         #                             # ells=ells,
#         #                             # dltt=dltt,
#         #                             # dlte=dlte,
#         #                             # dlee=dlee,
#         #                             # params=params,
#         #                             # lkl_setup=lkl_setup),
#         #                             # range(10))
#         # pool.close()
#         # print('r:',r)
#         # exit(0)
#         # print('dim:',dim)
#         # for i in range(lkl_setup.sp.n_specs):
#
#
#         if i<3:
#             spec = 'TT'
#             band1 = {0:'95',1:'95',2:'150'}[i]
#             band2 = {0:'95',1:'150',2:'150'}[i]
#             c1 = params[f'cal_{band1}']
#             c2 = params[f'cal_{band2}']
#
#             cls[sel] = fgpower.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dltt,spec,params,lkl_setup) * c1 * c2
#
#         elif i>=3 and i<=6:
#             spec = 'TE'
#             band1 = {0:'95',1:'95',2:'150',3:'150'}[i-3]
#             band2 = {0:'95',1:'150',2:'95',3:'150'}[i-3]
#             c1 = params[f'cal_{band1}']
#             c2 = params[f'cal_{band2}']
#             y = params[f'yp_{band2}']
#
#             cls[sel] = fgpower.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlte,spec,params,lkl_setup) * c1 * c2 * y
#
#         else:
#             spec = 'EE'
#             band1 = {0:'95',1:'95',2:'150'}[i-7]
#             band2 = {0:'95',1:'150',2:'150'}[i-7]
#             c1 = params[f'cal_{band1}']
#             c2 = params[f'cal_{band2}']
#             y1 = params[f'yp_{band1}']
#             y2 = params[f'yp_{band2}']
#
#             cls[sel] = fgpower.get_coadd_power(coadd_data[spec][(band1,band2)],bbl[i],ells,dlee,spec,params,lkl_setup) * c1 * c2 * y1 * y2
#
#
#
#     fgpower.cache['CIB'] = {}
#     return cls
#
#
#
#
#
# class StevePower(object):
#     def __init__(self,froot,flux,infval=1e10,tt_lmin=600,tt_lmax=None):
#         # print('doing stevepower')
#         spec=np.loadtxt(f"{froot}coadd_cl_{flux}_data_200124.txt")
#         cov =np.loadtxt(f'{froot}coadd_cov_{flux}_200519.txt')
#         bbl = np.loadtxt(f'{froot}coadd_bpwf_{flux}_191127_lmin2.txt')
#
#         if flux=='15mJy':
#             lregion = 'deep'
#         elif flux=='100mJy':
#             lregion = 'wide'
#         self.leak_95,self.leak_150 = np.loadtxt(f'{froot}leak_TE_{lregion}_200519.txt',usecols=[1,2],unpack=True)
#         # print('bbl shape:',np.shape(bbl))
#         # exit(0)
#         self.n_specs = 10
#         self.bbl =bbl.reshape((10,52,7924))
#         self.spec = spec[:520]
#         self.cov = cov[:520,:520]
#         nbin = 52
#         self.n_bins = nbin
#         self.ells = np.arange(2,7924+2)  # ell max of the full window functions = 7924
#         rells = np.repeat(self.ells[None],10,axis=0)
#         self.ls = self.bin(rells)
#
#         if tt_lmin is not None:
#             n = 3
#             ids = []
#             ids = np.argwhere(self.ls<tt_lmin)[:,0]
#             ids = ids[ids<nbin*3]
#             self.cov[:,ids] = 0
#             self.cov[ids,:] = 0
#             self.cov[ids,ids] = infval
#             # print('cov')
#             # print(cov)
#
#         if tt_lmax is not None:
#             n = 3
#             ids = []
#             ids = np.argwhere(self.ls>tt_lmax)[:,0]
#             ids = ids[ids<nbin*3]
#             self.cov[:,ids] = 0
#             self.cov[ids,:] = 0
#             self.cov[ids,ids] = infval
#             # print('cov')
#             # print(cov)
#
#         self.cinv = np.linalg.inv(self.cov)
#
#         if save_coadd_data == True:
#             band1 = '95'
#             band2 = '95'
#             flux = '100mJy'
#             path_root = None
#             spec = 'TT'
#
#             save_coadd_matrix(spec,band1,band2,flux,path_root)
#
#
#         fband1 = []
#         fband2 = []
#         for i in range(10):
#             if i<3:
#                 fband1.append({0:'T095',1:'T095',2:'T150'}[i])
#                 fband2.append({0:'T095',1:'T150',2:'T150'}[i])
#             elif i>=3 and i<=6:
#                 fband1.append({0:'T095',1:'T095',2:'T150',3:'T150'}[i-3])
#                 fband2.append({0:'E095',1:'E150',2:'E095',3:'E150'}[i-3])
#             else:
#                 fband1.append({0:'E095',1:'E095',2:'E150'}[i-7])
#                 fband2.append({0:'E095',1:'E150',2:'E150'}[i-7])
#         self.fband1 = fband1
#         self.fband2 = fband2
#
#         self.rfband1 =  np.repeat(self.fband1,nbin)
#         self.rfband2 =  np.repeat(self.fband2,nbin)
#         # print('rfband1 act:')
#         # print(self.rfband1)
#         # exit(0)
#
#
#     def bin(self,dls):
#         # print('bbl in bin, size : ',len(self.bbl[0,0,:]) )
#         # print(self.bbl[0,0,:])
#         bdl = np.einsum('...k,...k',self.bbl,dls[:,None,:])
#         return bdl.reshape(-1)
#
#     def select(self,bls,spec,band1,band2,shift=52):
#         I = {'tt':0,'te':3,'ee':7}
#         i = { 'tt':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2},
#               'te':{('95','95'): 0,('95','150'): 1,('150','95'): 2,('150','150'): 3},
#               'ee':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2} }
#         mind = i[spec][(band1,band2)]
#         sel = np.s_[(I[spec]+mind)*shift:(I[spec]+mind+1)*shift]
#         if bls.ndim==1: return bls[sel]
#         elif bls.ndim==2: return bls[sel,sel]
#         else: raise ValueError
#
#
#     # exit()
#
#
#
# class StevePower_extended(object):
#     def __init__(self,data_root,flux,infval=1e10,tt_lmin=600,tt_lmax=None):
#         # data_root = path_to_data + '/act_planck_data_210328/'
#         specs = ['f090xf090','f090xf100','f090xf143','f090xf150',
#          'f090xf217','f090xf353','f090xf545','f100xf100',
#          'f100xf143','f143xf143','f100xf150','f143xf150',
#          'f150xf150','f150xf217','f150xf353','f150xf545',
#          'f100xf217','f143xf217','f217xf217','f100xf353',
#          'f143xf353','f217xf353','f353xf353','f100xf545',
#          'f143xf545','f217xf545','f353xf545','f545xf545']
#         specx_l = ['TT']
#
#         freqs_asked = []
#         fband1 = []
#         fband2 = []
#         for spec in specs:
#             comp1 = spec.split('x')[0]
#             comp2 = spec.split('x')[1]
#             comp1 = comp1.replace('f', '')
#             comp2 = comp2.replace('f', '')
#             fband1.append(comp1)
#             fband2.append(comp2)
#             # print(comp1,comp2)
#             if comp1 not in freqs_asked:
#                 freqs_asked.append(comp1)
#             if comp2 not in freqs_asked:
#                 freqs_asked.append(comp2)
#         freqs_asked.sort()
#         # print('fband1: ')
#         # print(fband1)
#         # print('fband2: ')
#         # print(fband2)
#         self.fband1 = fband1
#         self.fband2 = fband2
#         # print('frequency list used in the analysis: ')
#         # print(freqs_asked)
#         self.cfreqs_list = freqs_asked
#
#         if flux == '15mJy':
#             rfroot = 'deep56'
#         if flux == '100mJy':
#             rfroot = 'boss'
#
#         spec = np.load(data_root+f'{rfroot}_all_ps_mean_C_ell_data_210327.npy')
#         cov = np.load(data_root+f'{rfroot}_all_ps_Cov_from_coadd_ps_210327.npy')
#         covx = np.load(data_root+f'{rfroot}_all_covmat_anal_210327.npy')
#         bbl = np.load(data_root+f'{rfroot}_bpwf_210327.npy')
#
#         l_min = 2
#         # print('bbl shape:',np.shape(bbl))
#         n_specs = len(specs)
#         # print('n_specs: ',n_specs)
#         n_bins = int(len(spec)/n_specs)
#         # print('n_bins: ',n_bins)
#         # exit(0)
#         n_ells = np.shape(bbl)[1]  # ell max of the full window functions = 3926
#         # print('n_ells: ',n_ells)
#         n_ells = n_ells-l_min
#         bbl_2 = np.zeros((n_bins*n_specs,n_ells))
#         for i in range(n_bins*n_specs):
#             bbl_2[i,:] = np.delete(bbl[i,:],[0,1])
#         # print(len(bbl_2[0,:]))
#
#
#         self.n_specs = n_specs
#
#
#         self.n_bins = n_bins
#
#         # doing coadd_data
#         #if self.bandpass:
#         if save_coadd_data_extended:
#             from scipy.linalg import block_diag
#             import pandas as pd
#             if flux=='15mJy':
#                 regions = ['deep56']
#             elif flux=='100mJy':
#                 regions = ['boss'] + [f'advact_window{x}' for x in range(6)]
#             else:
#                 raise ValueError
#             # order = spec
#             # print('starting df for region:',regions)
#             region = regions[0]
#             specx = specx_l[0]
#             order = np.load(data_root+f'{rfroot}_all_C_ell_data_order_190918.npy')
#             df = pd.DataFrame(order,columns=['t1','t2','region','s1','s2','a1','a2'])#.stack().str.decode('utf-8').unstack()
#             # print(df)
#             def rmap(r):
#                 if r[:6]=='advact': return 'advact'
#                 else: return r
#             # print('restricting df')
#             df = df[(df.t1==specx[0]) & (df.t2==specx[1]) & (df.region==rmap(region)+"_")]
#             # df = df[(df.t1==spec[0]) & (df.t2==spec[1]) & (df.region==rmap(region)+"_")]
#             # print(df)
#             # print('  ')
#             # print('  ')
#             for ib in range(28):
#                 # print('  ')
#                 # print('  ')
#
#                 band1 = fband1[ib]
#                 band2 = fband2[ib]
#
#                 arrays = []
#                 barrays = []
#                 icovs = []
#                 nbin = self.n_bins+3
#                 # print('ib b1, b2, nbin:',ib,band1,band2,nbin)
#                 rbin = 3
#                 # normallly here there is loop over region
#                 ibx = 0
#                 for index, row in df.iterrows():
#                     # print('id,row:',index,row)
#                     b1 = get_band_extended(row.a1)
#                     b2 = get_band_extended(row.a2)
#                     # print('b1,b2:',b1,b2)
#
#                     if (b1==band1 and b2==band2) or (b1==band2 and b2==band1):
#                         #print('row:',row)
#                         # print('rmap(region):',index,ibx,rmap(region))
#                         # print('rmap b1,b2:',b1,b2)
#                         # print('row.s1:',row.s1)
#                         # print('row.s2:',row.s2)
#                         # print('row.a1:',row.a1)
#                         # print('row.a2:',row.a2)
#                         # print(' ')
#                         # print('#####')
#                         arrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
#                         barrays.append((index,rmap(region),row.s1,row.s2,row.a1,row.a2))
#                         ibx += 1
#                 adf = pd.DataFrame(arrays,columns = ['i','r','s1','s2','a1','a2'])
#                 ids = adf.i.to_numpy()
#                 oids = []
#                 for ind in ids:
#                     oids = oids + list(range(ind*nbin+rbin,(ind+1)*nbin))
#                 # print('oids:',oids)
#                 # print('scov:',np.shape(cov))
#                 # print('soids:',np.shape(oids))
#                 # try:
#                 ocov = covx[oids,:][:,oids]
#
#                 #exit(0)
#                 # print('socov:',np.shape(ocov))
#                 icovs.append(np.linalg.inv(ocov))
#                 # print('sicovs:',np.shape(icovs))
#                 icov = block_diag(*icovs)
#                 # print('sicov:',np.shape(icov))
#                 # print('barrays:',barrays)
#                 nspec = 1
#                 nbins = self.n_bins
#                 norig = len(barrays)
#                 # print('norig:',norig)
#                 # print('barrays:',barrays)
#                 N_spec = nbins*nspec
#                 # print('N_spec:',N_spec)
#                 pmat = np.identity(N_spec)
#                 Pmat = np.identity(N_spec)
#                 for i in range(1,norig):
#                     Pmat = np.append(Pmat,pmat,axis=1)
#                 # print('sPmat:',np.shape(Pmat))
#                 icov_ibin = np.linalg.inv(np.dot(Pmat,np.dot(icov,Pmat.T)))
#                 barrays = np.array(barrays,dtype=[('i','i8'),('r','U32'),('s1','U32'),('s2','U32'),('a1','U32'),('a2','U32')])
#                 # print('barrays:',barrays)
#                 np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_icov.txt',icov)
#                 np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_icov_ibin.txt',icov_ibin)
#                 np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_pmat.txt',Pmat)
#                 np.savetxt(data_root+f'{rfroot}_{specx}_{band1}_{band2}_{flux}_arrays.txt', barrays,fmt=['%d','%s','%s','%s','%s','%s'])
#
#                 # print('ib2:',ib,band1,band2)
#             #
#         #exit(0)
#
#
#
#
#
#
#         self.bbl = bbl_2.reshape((n_specs,n_bins,n_ells))
#         self.spec = spec[:,1]
#         self.cov = cov
#         # print('shape cov : ', np.shape(self.cov))
#         nbin = n_bins
#         #self.ells = np.arange(2,n_ells+2)
#         self.ells = np.arange(l_min,n_ells+2)
#         # self.ells = np.arange(2,7924+2)
#         # self.ells = spec[:,0]
#         #rells = np.repeat(self.ells[None],n_specs,axis=0)
#         #print(np.shape(rells))
#         #self.ls = self.bin(rells)
#         self.ls = spec[:,0]
#
#         # conversion factor to bplike normalisations, i.e., dl's to cl's:
#         fac = self.ls*(self.ls+1.)/2./np.pi
#         self.spec = self.spec/fac
#         # print('shape spec : ', np.shape(self.spec))
#         # print('shape fac : ', np.shape(fac))
#         self.cov = self.cov/fac**2.
#
#
#
#         if tt_lmin is not None:
#             # n = 3
#             ids = []
#             ids = np.argwhere(self.ls<tt_lmin)[:,0]
#             # print('ls: ', self.ls)
#             # print(len(self.ls))
#             # print('nbins: ',nbin)
#             # print('ids ls<tt_lmin: ', ids)
#             # print('fband1 : ',self.fband1)
#             rfband1 = np.repeat(self.fband1,nbin)
#             rfband2 = np.repeat(self.fband2,nbin)
#             self.rfband1 =  rfband1
#             self.rfband2 =  rfband2
#             # print(len(rfband2))
#             # print(type(rfband2[0]))
#             # cd_act =
#             # set cov to infal (ignore points) where l is smaller than tt_lmin=600 (default) for the ACT bands
#             ids_act = np.argwhere((self.ls<tt_lmin) & ((rfband1 == '090') | (rfband1 == '150') | (rfband2 == '090') | (rfband2 == '150')))[:,0]
#             # print('idsact : ',ids_act)
#             # exit(0)
#             ids = ids_act
#             self.cov[:,ids] = 0
#             self.cov[ids,:] = 0
#             self.cov[ids,ids] = infval
#             # print('cov')
#             # print(cov)
#         # print('setting inf in cov where beam too small')
#
#         beam_dict_f100 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F100.txt')
#         beam_dict_f143 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F143.txt')
#         beam_dict_f217 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F217.txt')
#         beam_dict_f353 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F353.txt')
#         beam_dict_f545 = np.loadtxt(data_root + 'HFI_BEAM_resave_210414_F545.txt')
#
#         beam_cut_off = 0.1
#         lmax_beam_cutoff = {}
#         lmax_100 = beam_dict_f100[:,0][beam_dict_f100[:,1]>beam_cut_off].max()
#         lmax_143 = beam_dict_f100[:,0][beam_dict_f143[:,1]>beam_cut_off].max()
#         lmax_217 = beam_dict_f100[:,0][beam_dict_f217[:,1]>beam_cut_off].max()
#         lmax_353 = beam_dict_f100[:,0][beam_dict_f353[:,1]>beam_cut_off].max()
#         lmax_545 = beam_dict_f100[:,0][beam_dict_f545[:,1]>beam_cut_off].max()
#
#         lmax_090 = infval
#         lmax_150 = infval
#         lmax_beam_cutoff['090'] = lmax_090
#         lmax_beam_cutoff['150'] = lmax_150
#         lmax_beam_cutoff['100'] = lmax_100
#         lmax_beam_cutoff['143'] = lmax_143
#         lmax_beam_cutoff['217'] = lmax_217
#         lmax_beam_cutoff['353'] = lmax_353
#         lmax_beam_cutoff['545'] = lmax_545
#
#         # print(lmax_090,lmax_150,lmax_100,lmax_143,lmax_217,lmax_353,lmax_545)
#         # cov has dimension 1344
#         # this is n_bin (48) times n_spec (28)
#         # print(np.shape(self.cov))
#         # print('fband1 : ',self.fband1)
#         # print('fband2 : ',self.fband2)
#         lmax_order_list = []
#
#         for (fb1,fb2) in zip(self.fband1,self.fband2):
#             lmax_order_list.append(min(lmax_beam_cutoff[fb1],lmax_beam_cutoff[fb2]))
#
#         # print('lmax_order_list:',lmax_order_list)
#
#
#
#         # rfband1 = np.repeat(self.fband1,nbin)
#         # print('repeated fband1:',rfband1)
#         # rfband2 = np.repeat(self.fband2,nbin)
#         rlmax_order_list = np.repeat(lmax_order_list,nbin)
#         # print(rlmax_order_list)
#
#         ids_cutoff = np.argwhere((self.ls>rlmax_order_list))[:,0]
#         # print(self.ls)
#         # print(ids_cutoff)
#         ids = ids_cutoff
#         self.cov[:,ids] = 0
#         self.cov[ids,:] = 0
#         self.cov[ids,ids] = infval
#         # exit(0)
#         # if tt_lmax is not None:
#         #     n = 3
#         #     ids = []
#         #     ids = np.argwhere(self.ls>tt_lmax)[:,0]
#         #     ids = ids[ids<nbin*3]
#         #     self.cov[:,ids] = 0
#         #     self.cov[ids,:] = 0
#         #     self.cov[ids,ids] = infval
#
#         self.cinv = np.linalg.inv(self.cov)
#
#     def bin(self,dls):
#         # print('bbl in bin, size : ',len(self.bbl[0,0,:]) )
#         # print(self.bbl[0,0,:])
#         bdl = np.einsum('...k,...k',self.bbl,dls[:,None,:])
#         return bdl.reshape(-1)
#
#     #
#     # def select(self,bls,spec,band1,band2,shift=52):
#     #     I = {'tt':0,'te':3,'ee':7}
#     #     i = { 'tt':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2},
#     #           'te':{('95','95'): 0,('95','150'): 1,('150','95'): 2,('150','150'): 3},
#     #           'ee':{('95','95'): 0,('95','150'): 1,('150','95'): 1,('150','150'): 2} }
#     #     mind = i[spec][(band1,band2)]
#     #     sel = np.s_[(I[spec]+mind)*shift:(I[spec]+mind+1)*shift]
#     #     if bls.ndim==1: return bls[sel]
#     #     elif bls.ndim==2: return bls[sel,sel]
#     #     else: raise ValueError
#


class act_pylike_extended_act_only_TTTEEE(InstallableLikelihood):


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
            "a_g_tt_15", # TT Galactic dust at ell=500
            "a_g_tt_100", # TT Galactic dust at ell=500
            "a_g_te_15", # TE Galactic dust at ell=500
            "a_g_te_100", # TE Galactic dust at ell=500
            "a_g_ee_15", # EE Galactic dust at ell=500
            "a_g_ee_100", # EE Galactic dust at ell=500
            "a_s_te", # TE Synchrotron at ell=500
            "a_s_ee"] # EE Synchrotron at ell=500
        # "cal_95",
        # "cal_150",
        # "yp_95",
        # "yp_150"]
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
        # file = resource_filename("bplike","act_pylike_extended_full.yaml")
        # with open(file) as f:
        #     act_pylike_extended_full = yaml.load(f, Loader=yaml.FullLoader)
        # print('act_pylike_extended_full')
        # print(act_pylike_extended_full)
        # exit(0)
        # Load path_params from yaml file
        if self.use_act_planck == 'no':
            # self.l_max = 6000
            self.l_max = 6051
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
            self.l_max = 3924
            # self.l_max = 6051
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
        # new_file = file.replace('_full', '')
        # print('new_file')
        # print(new_file)
        # with open(new_file, 'w') as f:
        #     yaml.dump(act_pylike_extended_full, f)
        # # exit(0)




        self.expected_params = list(np.concatenate((self.expected_params,cal_yp)))
        # print('expected params: ',self.expected_params)
        # exit(0)
        self.bands = self.aparams['bands']


        # Read data
        # print('preparing data')
        self.prepare_data()

        # State requisites to the theory code
        # if self.use_act_planck == 'yes':
        #     self.requested_cls = ["tt"]
        # else:
        #     self.requested_cls = ["tt", "te", "ee"]






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
        # print('input params: ',self.input_params)

        # differences = are_different_params_lists(
        #     self.input_params, self.expected_params,
        #     name_A="given", name_B="expected")
        # if differences:
        #     # self.input_params = self.expected_params
        #     raise LoggedError(
        #         self.log, "Configuration error in parameters: %r.",
        #         differences)

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
        # return 0
        # print('doing logp')

        cl = self.theory.get_Cl(ell_factor=True)
        # print("cl's: ",cl)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        # print('params:',params_values)
        # print('##############')
        # print(' ')

        # print('doing loglike')
        # print('cls 0:10:', cl['tt'][:10] )
        comps = ['tot','primary','tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']


        # print('loglike getting power spectra')
        ps = self._get_power_spectra(cl, lkl_setup = self, **params_values)
        # print('loglike done getting power spectra')
        ps_vec = ps['tot']
        # ps_vec = ps['primary']
        # print('model')
        # for idb in range(52):
        #     print(idb,ps_vec[:52][idb])
        # print('data')


        # ps_vec_galdust = ps['galdust']
        # ps_vec_primary = ps['primary']
        # print('shape ls: ',np.shape(self.sp.ls))
        # print('shape ps_vec : ', np.shape(ps_vec))
        # print('shape self.sp.spec : ', np.shape(self.sp.spec))
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
            if save_theory_data_spectra:
                np.save(path_to_output+'/ls_theory_'+self.flux+bps+'act_planck.npy',ls_theory)

                for comp in comps:
                    np.save(path_to_output+'/dls_theory_'+comp+'_'+self.flux+bps+'act_planck.npy',ps[comp])



        elif self.use_act_planck == 'no':
            # print('ps_vec : ', ps_vec[:n_bins])
            # print('self.sp.spec : ', self.sp.spec[:n_bins])
            dls_theory = ps_vec
            ls_theory = self.sp.ls

            # print('len ls ps:',len(ls_theory),len(dls_theory))

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




            # print(ids_leak)
            #
            # exit(0)


            delta = self.sp.spec - ps_vec
            if save_theory_data_spectra:
                np.save(path_to_output+'/ls_theory_'+self.flux+bps+'act_only.npy',ls_theory)
                for comp in comps:
                    np.save(path_to_output+'/dls_theory_'+comp+'_'+self.flux+bps+'act_only.npy',ps[comp]*fac)


            # np.save(path_to_output+'/dls_theory_galdust_'+self.flux+bps+'act_only.npy',ps_vec_galdust*fac)
            # np.save(path_to_output+'/dls_theory_primary_'+self.flux+bps+'act_only.npy',ps_vec_primary*fac)


        #
        # for idb in range(520):
        #     print("%s %s %d\t%.10e\t%.10e\t%.10e\t%.10e"%(self.sp.rfband1[idb],self.sp.rfband2[idb],idb+1,self.sp.cinv[idb,idb],delta[idb],self.sp.spec[idb],dls_theory[idb]))
        # # exit(0)


        logp = -0.5 * np.dot(delta,np.dot(self.sp.cinv,delta))
        if self.theory_debug is not None:
            print('[debug] logp: ',logp)
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

                # beam_dict['s12_pa0_f100'] = data_root + 'HFI_BEAM_F100.txt'
                # beam_dict['s12_pa0_f143'] = data_root + 'HFI_BEAM_F143.txt'
                # beam_dict['s12_pa0_f217'] = data_root + 'HFI_BEAM_F217.txt'
                # beam_dict['s12_pa0_f353'] = data_root + 'HFI_BEAM_F353.txt'
                # beam_dict['s12_pa0_f545'] = data_root + 'HFI_BEAM_F545.txt'




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
        # print('cls:',cl)

        if self.theory_debug is not None:
            print('[debug] theory debug:',self.theory_debug)
            # ells,cltt,clee,clte = np.loadtxt(self.theory_debug,usecols=[0,1,2,4],unpack=True) # mat's debig
            ells,cltt,clte,clee = np.loadtxt(self.theory_debug,usecols=[0,1,2,3],unpack=True) # boris's debug
            # print('ell0,ell1:',ells[0],ells[1],ells[:self.l_max])

            assert ells[0] == 2
            assert ells[1] == 3
            cl = {}
            # cl['ell'] = np.zeros(2+self.l_max+50)
            # cl['tt'] = np.zeros(2+self.l_max+50)
            # cl['te'] = np.zeros(2+self.l_max+50)
            # cl['ee'] = np.zeros(2+self.l_max+50)
            # cl['ell'][1] = 1
            # cl['ell'][2:] = ells[:self.l_max+50]
            # cl['tt'][2:] = cltt[:self.l_max+50]
            # cl['te'][2:] = clte[:self.l_max+50]
            # cl['ee'][2:] = clee[:self.l_max+50]
            # 6051
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

            # cl['ell'] = np.zeros(self.l_max)
            # cl['tt'] = np.zeros(self.l_max)
            # cl['te'] = np.zeros(self.l_max)
            # cl['ee'] = np.zeros(self.l_max)
            #
            # cl['ell'][1] = 1
            # cl['ell'][2:] = ells[2:self.l_max]
            # cl['tt'][2:] = cltt[2:self.l_max]
            # cl['te'][2:] = clte[2:self.l_max]
            # cl['ee'][2:] = clee[2:self.l_max]
            # cl['ell'] = np.zeros(6051)
            # cl['tt'] = np.zeros(6051)
            # cl['te'] = np.zeros(6051)
            # cl['ee'] = np.zeros(6051)
            # cl['ell'][1] = 1
            # cl['ell'][2:] = ells[:6051]
            # cl['tt'][2:] = cltt[:6051]
            # cl['te'][2:] = clte[:6051]
            # cl['ee'][2:] = clee[:6051]
        # print('cl:',cl)
        # save some cls to do debugging fun
        # np.savetxt(path_to_output+'/cls_model.txt',np.c_[cl['ell'],cl['tt'],cl['te'],cl['ee']])
        # exit(0)
        # exit(0)

        fgdict =    {k: params_values[k] for k in self.expected_params}
        fgdict.update(self.fparams)
        nells_camb = cl['ell'].size

        nells = self.sp.ells.size
        # print('nells_camb (dim of cl["ell"]): ', nells_camb)
        # print('nells (dim of cl["ell"]): ', nells_camb)
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


        # ptt[l_min:nells_camb] = cl['tt'][l_min:]
        # pte[l_min:nells_camb] = cl['te'][l_min:]
        # pee[l_min:nells_camb] = cl['ee'][l_min:]

        # ptt[l_min:nells_camb] = cl['tt'][l_min:nells_camb]
        # pte[l_min:nells_camb] = cl['te'][l_min:nells_camb]
        # pee[l_min:nells_camb] = cl['ee'][l_min:nells_camb]


        # print('lencl["tt"][l_min:] :', len(cl['tt'][l_min:]))
        # print('lenptt[l_min:nells_camb] :', len(ptt[l_min:nells_camb]))

        # exit(0)
        if lkl_setup.use_act_planck == 'no':
            ptt[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
            pte[l_min:nells_camb] = cl['te'][l_min:len(ptt[l_min:nells_camb])+l_min]
            pee[l_min:nells_camb] = cl['ee'][l_min:len(ptt[l_min:nells_camb])+l_min]
        elif lkl_setup.use_act_planck == 'yes':
            ptt[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
            pte[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
            pee[l_min:nells_camb] = cl['tt'][l_min:len(ptt[l_min:nells_camb])+l_min]
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
            # dim = lkl_setup.sp.n_bins*lkl_setup.sp.n_specs
            # clsp = np.zeros((dim,))
            # fgpow = []
            # for i in range(lkl_setup.sp.n_specs):
            #     fgpowp = get_theory_bandpassed_parallel(i,self.fgpower,
            #                                                 self.coadd_data,
            #                                                 self.sp.ells,
            #                                                 self.sp.bbl,
            #                                                 ptt[l_min:],
            #                                                 pte[l_min:],
            #                                                 pee[l_min:],
            #                                                 fgdict,
            #                                                 lmax=self.l_max,
            #                                                 lkl_setup = lkl_setup)
            #     fgpow.append(fgpowp)
            #     print('fgpow:',np.shape(fgpowp),fgpowp)
            #
            #
            # pool = multiprocessing.Pool()
            # fgstructure = self.fgpower
            # fn = functools.partial(get_theory_bandpassed_parallel_2,
            #                        coadd_data = fgstructure)
            #
            # r = pool.map(fn,range(lkl_setup.sp.n_specs))
            # pool.close()
            # print('r:',r)
            # exit(0)
            # pool = multiprocessing.Pool()
            # fn = functools.partial(get_theory_bandpassed_parallel,
            #                        fgpower = self.fgpower,
            #                        coadd_data = self.coadd_data,
            #                        ells = self.sp.ells,
            #                        bbl = self.sp.bbl,
            #                        ptt = ptt[l_min:],
            #                        pte = pte[l_min:],
            #                        pee = pee[l_min:],
            #                        fgdict = fgdict,
            #                        lmax=self.l_max,
            #                        lkl_setup = lkl_setup)
            # r = pool.map(fn,range(lkl_setup.sp.n_specs))
            # pool.close()
            # print('r:',r)
            # exit(0)
            #
            # for i in range(lkl_setup.sp.n_specs):
            #     sel = np.s_[i*lkl_setup.sp.n_bins:(i+1)*lkl_setup.sp.n_bins]
            #     clsp[sel] = fgpow[i][sel]
            # fpower['tot'] = clsp
            # print('fpower:',fpower)
            # # exit(0)
            #
            # if self.theory_debug is not None:
            #     print('[debug] time for tot: ', time.time() - start)
            # for comp in comps:
            #     if self.theory_debug is not None:
            #         print('[debug] comp = ', comp)
            #         start = time.time()
            #     fpower[comp] = self.fgpower.get_theory_bandpassed_comp(self.coadd_data,
            #                                             self.sp.ells,
            #                                             self.sp.bbl,
            #                                             ptt[l_min:],
            #                                             pte[l_min:],
            #                                             pee[l_min:],
            #                                             fgdict,
            #                                             lmax=self.l_max,
            #                                             lkl_setup = lkl_setup,
            #                                             comp = comp)
            #     if self.theory_debug is not None:
            #         print('[debug] time for comp: ', time.time() - start)

            return fpower


        else:
            # print('doing theory no bandpass')
            # print('sp.ells: ',self.sp.ells)
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
            if save_theory_data_spectra:
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

# class act15_act_TT_plus_planck_TT(act_pylike_extended):
#     flux = '15mJy'
#     use_act_planck = 'yes'
#
# class act100_act_TT_plus_planck_TT(act_pylike_extended):
#     flux = '100mJy'
#     use_act_planck = 'yes'