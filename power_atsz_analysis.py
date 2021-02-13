# I haven't made bplike into a package yet, so I run everything in the same dir.
# The stuff you need is all in the fg.py module; note that it depends on tileC.
import fg
import numpy as np
import os,sys
import argparse
import ast




def config_from_yaml(filename):
    import yaml
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


def parse_args(args):
    # Parse command line
    parser = argparse.ArgumentParser(description='Do a thing.')
    parser.add_argument("--output_dir", type=str,default='output_dir',help='path to output_dir for bplike spectra')
    parser.add_argument("--specs", type=str,default='specs',help='spectra to compute')
    parser.add_argument("--bplike_params_dict", type=str,default='bplike_params_dict',help='bplike_params_dict')
    args = parser.parse_args()
    return args

def main():
    args = parse_args(sys.argv[1:])
    # print(args.output_dir)
    # print(args.specs)
    # exit(0)
    output_dir = args.output_dir
    specs = ast.literal_eval(args.specs)

    sz_temp_file = "data/cl_tsz_150_bat.dat"
    sz_x_cib_temp_file = "data/sz_x_cib_template.dat"
    ksz_temp_file = "data/cl_ksz_bat.dat"

    bplike_params_dict = ast.literal_eval(args.bplike_params_dict)


    # Internal identifier for what radio source parameter to vary
    flux_cut = bplike_params_dict['flux_cut'] #'15mJy'
    # flux_cut = '15mJy'


    # Load fiducial foreground parameters
    params = config_from_yaml("fg_defaults.yml")

    for key,val in bplike_params_dict.items():
        if key in list(params.keys()):
            params[key] = val


    # Multipoles to evaluate at
    ells = np.arange(2,10000)

    # Initialize the power generator
    comps = ['tsz','cibc','cibp','tsz_x_cib','radio','ksz']#,'ksz','cibc','cibp','tsz_x_cib','radio','galdust','galsyn']
    fp = fg.ForegroundPowers(params,ells,
                             sz_temp_file,ksz_temp_file,sz_x_cib_temp_file,flux_cut,
                             arrays=None,bp_file_dict=None,beam_file_dict=None,cfreq_dict=None, # Only used with bandpasses
                             comps = comps) # Components to initialize

    freqs_asked = []

    for spec in specs:
        freq1 = int(spec.split('x')[0])
        freq2 = int(spec.split('x')[0])
        freqt = (freq1,freq2)
        freqs_asked.append(freqt)
    freqs_asked = tuple(freqs_asked)



    for freqs in freqs_asked:
        freq1 = freqs[0] # Ghz
        freq2 = freqs[1]
        efreq1 = {'dust':freq1,'tsz':freq1,'syn':freq1} # All components have the same effective frequency
        efreq2 = {'dust':freq2,'tsz':freq2,'syn':freq2}



        # Now get D_ell_foreground = C_ell * l (l+1) / 2 / pi
        dltt = []
        dltt.append(ells)
        dltt_tot = fp.get_power("TT",
                            comps, # components to include in sum
                            params,
                            eff_freq_ghz1=efreq1,array1=None,
                            eff_freq_ghz2=efreq2,array2=None,lmax=None)
        dltt.append(dltt_tot)
        for cp in comps:

            dltt_cp = fp.get_power("TT",
                                [cp], # components to include in sum
                                params,
                                eff_freq_ghz1=efreq1,array1=None,
                                eff_freq_ghz2=efreq2,array2=None,lmax=None)
            dltt.append(dltt_cp)

        np.savetxt(output_dir+"spectra_l_dltt_tot_tsz_cibc_cibp_rs_tszxcib_ksz_"+str(freq1)+"_"+str(freq2)+".txt",
                  np.transpose(dltt))



if __name__ == "__main__":
    main()
