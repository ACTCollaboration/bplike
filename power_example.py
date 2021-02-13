# I haven't made bplike into a package yet, so I run everything in the same dir.
# The stuff you need is all in the fg.py module; note that it depends on tileC.
import fg
import numpy as np

output_dir = '/Users/boris/Work/CLASS-SZ/SO-SZ/bplike/output/'

def config_from_yaml(filename):
    import yaml
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


sz_temp_file = "data/cl_tsz_150_bat.dat"
sz_x_cib_temp_file = "data/sz_x_cib_template.dat"
ksz_temp_file = "data/cl_ksz_bat.dat"

# Internal identifier for what radio source parameter to vary
flux_cut = '15mJy'

# Load fiducial foreground parameters
params = config_from_yaml("fg_defaults.yml")

# Multipoles to evaluate at
ells = np.arange(2,10000)

# Initialize the power generator
#comps = ['tsz','ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']
comps = ['cibc']#,'ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']
fp = fg.ForegroundPowers(params,ells,
                         sz_temp_file,ksz_temp_file,sz_x_cib_temp_file,flux_cut,
                         arrays=None,bp_file_dict=None,beam_file_dict=None,cfreq_dict=None, # Only used with bandpasses
                         comps = comps) # Components to initialize

freq1 = 143. # Ghz
freq2 = 143.
efreq1 = {'dust':freq1,'tsz':freq1,'syn':freq1} # All components have the same effective frequency
efreq2 = {'dust':freq2,'tsz':freq2,'syn':freq2}

# Now get D_ell_foreground = C_ell * l (l+1) / 2 / pi
dltt = fp.get_power("TT",
                    comps, # components to include in sum
                    params,
                    eff_freq_ghz1=efreq1,array1=None,
                    eff_freq_ghz2=efreq2,array2=None,lmax=None)
print(np.shape(dltt))

dlee = fp.get_power("EE",
                    comps, # components to include in sum
                    params,
                    eff_freq_ghz1=efreq1,array1=None,
                    eff_freq_ghz2=efreq2,array2=None,lmax=None)
print(np.shape(dlee))
dlte = fp.get_power("TE",
                    comps, # components to include in sum
                    params,
                    eff_freq_ghz1=efreq1,array1=None,
                    eff_freq_ghz2=efreq2,array2=None,lmax=None)

print(np.shape(dlte))
try:
    np.savetxt(output_dir+"spectra_l_dltt_dlee_dlte_"+str(freq1)+"_"+str(freq2)+".txt",np.c_[ells,dltt,dlee,dlte])
except:
    np.savetxt(output_dir+"spectra_l_dltt_dlee_dlte_"+str(freq1)+"_"+str(freq2)+".txt",np.c_[ells,dltt])
