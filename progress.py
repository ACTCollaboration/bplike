from cobaya.samplers.mcmc import plot_progress
import numpy as np
import argparse
import yaml
import camb
from camb import model
import matplotlib.pyplot as plt
# from orphics import maps

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("--unblind", action='store_true',help='Unblind.')
parser.add_argument("--compressed", action='store_true',help='Compressed triangle plot.')
parser.add_argument("--skip-progress", action='store_true',help='Skip progress.')
parser.add_argument("--loga",     type=str,  default="logA",help="Sim ID.")
args = parser.parse_args()
print("Command line arguments are %s." % args)

version = args.version
logAName = args.loga
outpath = f"chains/{version}"

if not(args.skip_progress):
    plot_progress(outpath)
    plt.tight_layout()
    plt.savefig(f"progress_{version}.png")


import sys, os
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt

from getdist.mcsamples import loadMCSamples
samples = loadMCSamples(outpath,settings={'ignore_rows':0.3})

# covmat = samples.getCovMat()
# covmat.saveToFile("covmat.txt")
# sys.exit()

p = samples.getParams()

tests = samples.getConvergeTests(what=['GelmanRubin'])
print(tests)

pparams1 = [str(x.name) for x in samples.getParamNames().names]
pparams = list(pparams1)
for p in pparams1:
    if ('chi2' in p ) or ('minuslogprior' in p): pparams.remove(p)


pmap = \
       { \
         'aksz':'a_ksz',
         'atsz':'a_tsz',
         'amp_c':'a_c',
         'amp_d':'a_d',
         'amp_ps':'a_p_ee',
         'amp_s':'a_p_tt_15',
         'amp_sw':'a_p_tt_100',
         'amp_tps':'a_p_te',
         'beta_c': 'beta_CIB',
         'H0':'H0',
         'logA': 'logA',
         'ns': 'ns',
         'omegabh2':'ombh2',
         'omegach2':'omch2',
         'sigma8':'sigma8',
         'tau':'tau',
         'theta':'theta_MC_100',
         'xi':'xi',
         'yp1':'yp_95',
        'yp2':'yp_150'
     }

lmap = \
       { \
         'aksz':'A_{\\rm kSZ}',
         'atsz':'A_{\\rm tsz}',
         'amp_c':'A_{\\rm CIB}',
         'amp_d':'A_{\\rm CIB-poisson}',
         'amp_ps':'A_{\\rm EE-ps}',
         'amp_s':'A_{\\rm TT-ps-15mJy}',
         'amp_sw':'A_{\\rm TT-ps-100mJy}',
         'amp_tps':'A_{\\rm TE-ps}',
         'beta_c': '\\beta_{\\rm CIB}',
         'H0':'H_0',
         'logA': '{\\rm log}(A)',
         'ns': 'n_s',
         'omegabh2':'\\Omega_b h^2',
         'omegach2':'\\Omega_c h^2',
         'sigma8':'\\sigma_8',
         'tau':'\\tau',
         'theta':'\\theta_{\\rm MC-100}',
         'xi':'\\xi',
         'yp1':'y_{\\rm p-95}',
         'yp2':'y_{\\rm p-150}'
     }


print(pparams)
# from orphics import io
# for ip in pmap.keys():
#     p = pmap[ip]
#     print(p)
#     d = samples.get1DDensity(p)
#     ps,probs = np.loadtxt(f"plot_data/ACTPol_Feb24_p_{ip}.dat",unpack=True)
#     pl = io.Plotter(xlabel=f'${lmap[ip]}$',ylabel='P',xyscale='linlin')
#     pl.add(ps,probs)
#     pl.add(ps,d.Prob(ps),ls='--')
#     pl.done(f'prob_{version}_{ip}.png')


# ny = 4
# nx = 5
# fig, axs = plt.subplots(ny, nx,figsize=(20,16))
# keys = list(pmap.keys())
# print(len(keys))
# c = 0
# for i in range(ny):
#     for j in range(nx):
#         ip = keys[c]
#         p = pmap[ip]
#         d = samples.get1DDensity(p)
#         ps1,probs1 = np.loadtxt(f"plot_data/ACTPol_Feb24_p_{ip}.dat",unpack=True)
#         if ip=='amp_sw':
#             ps = np.linspace(21,25.5,1000)
#         elif ip=='amp_d':
#             ps = np.linspace(4,8,1000)
#         else:
#             ps = ps1

#         probs = maps.interp(ps1,probs1)(ps)
#         axs[i, j].plot(ps, probs)
#         axs[i,j].plot(ps,d.Prob(ps),ls='--')
#         axs[i,j].set_xlabel(f'${lmap[ip]}$', fontsize=14)

#         c = c + 1

# fig.savefig(f'fprob_{version}.png')
# sys.exit()

#sys.exit()



stats = samples.getMargeStats()
print(stats)

g = plots.get_subplot_plotter()


t = g.triangle_plot([samples], pparams, filled=True)


plt.savefig(f"triangle_{version}.png")


