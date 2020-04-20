from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
from tilec import fg as tfg
import yaml

fgroot = os.path.dirname(os.path.abspath(__file__)) +  f"/data/actpol_2f_full_s1316_2flux_fin/data/Fg/"

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
    return maps.interp(ls,cls)(ells)
