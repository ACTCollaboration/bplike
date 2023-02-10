# bplike

This is a band-passed likelihood code for CMB power spectrum analysis, currently implemented primarily with interfaces to the ACT DR4 measurement.

It is similar to the official likelihood for the ACT DR4 release, but with the following changes:
- includes (optional) scale-dependent bandpass corrections 
- it is re-written in Python 
- it interfaces with Cobaya

It reproduces the official results when run without the 
bandpass corrections.

It currently has dependencies that include soapack and tilec,
which should ideally be removed.

## Data

You can find the data files here on niagara:
```
/scratch/r/rbond/msyriac/data/depot/bplike/data
```

and on cori (NERSC):
```
/global/project/projectdirs/act/data/msyriac/bplike_data
```

You should symlink that directory to the root repository directory with the name `data`.


## Running chains with Cobaya

Simply wrap
```
cobaya-run act.yml
```

with your preferred OpenMP-MPI method. `act.yml` is an example YAML
file for the run with bandpass corrections. You can copy and modify it,
e.g. to change priors or add additional parameters. `act_baseline.yml`
contains an example for reproducing the official DR4 runs, i.e. without the
bandpass corrections.
