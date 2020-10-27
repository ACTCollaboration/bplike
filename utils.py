from __future__ import print_function
import numpy as np
import os,sys
from tilec import fg as tfg
import yaml
from scipy.interpolate import interp1d

fgroot = os.path.dirname(os.path.abspath(__file__)) +  f"/data/actpol_2f_full_s1316_2flux_fin/data/Fg/"


def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)


def config_from_yaml(filename):
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
