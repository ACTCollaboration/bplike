# bplike

This is a version of Erminia's multi-frequency likelihood
for the ACT DR4 release, but with the following changes:
- includes (optional) scale-dependent bandpass corrections 
- it is re-written in Python 
- it interfaces with Cobaya (use the `devel` branch)

It reproduces Erminia's results when run without the 
bandpass corrections.

It currently has dependencies that include soapack and tilec,
which should ideally be removed.

## Data

You can find the data files here on niagara:
```
/scratch/r/rbond/msyriac/data/depot/bplike/data
```
You should symlink that directory to the root repository directory.


## Running chains with Cobaya

Simply wrap
```
cobaya-run act.yml
```

with your preferred OpenMP-MPI method. `act.yml` is an example YAML
file for the run with bandpass corrections. You can copy and modify it,
e.g. to change priors or add additional parameters. `act_baseline.yml`
contains an example for reproducing Erminia's runs, i.e. without the
bandpass corrections.
