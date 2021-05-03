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
<<<<<<< HEAD
and on NERSC:

```
 /global/project/projectdirs/act/data/msyriac/bplike_data
 ```

You should download that directory and move it to bplike/data/.
Then adapt the path in bplike/config.py so that it points toward the bplike/data directory.
=======

and on cori (NERSC):
```
/global/project/projectdirs/act/data/msyriac/bplike_data
```

You should symlink that directory to the root repository directory with the name `data`.
>>>>>>> 0932af78273aa1a934c53e9d898c0894c6e204d5


## Running chains with Cobaya

Simply run
```
$ cobaya-run act_extended.yml -f
```
and adapt the yml file acccording to what you would like to do.
