from __future__ import print_function
import numpy as np
import os,sys
from tilec import fg as tfg
import yaml
from scipy.interpolate import interp1d
from pkg_resources import resource_filename
from .config import *

# fgroot = os.path.dirname(os.path.abspath(__file__)) +  f"/data/actpol_2f_full_s1316_2flux_fin/data/Fg/"

save_coadd_data = False
save_coadd_data_extended = False
# save_theory_data_spectra = False

fgroot = resource_filename("bplike","data/actpolfull_dr4.01/data/Fg/")

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

def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)


def config_from_yaml(filename):
    filename = resource_filename("bplike",filename)
    # print('config from :', filename)
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


def get_erminia_fg(ells,comp):
    if comp=='ksz':
        ls,dls = np.loadtxt(f"{fgroot}cl_ksz_bat.dat",unpack=True)
    elif comp=='tsz':
        ls,dls = np.loadtxt(f"{fgroot}cl_tsz_150_bat.dat",unpack=True)
        dls = dls / tfg.get_mix(150., 'tSZ')**2.


    with np.errstate(divide='ignore'):
        cls = dls*2.*np.pi/ls**2.
    cls[ls<1] = 0
    return interp(ls,cls)(ells)




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
