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

and on NERSC:

```
 /global/project/projectdirs/act/data/msyriac/bplike_data
 ```

You should download that directory and move it to bplike/data/.
Data path are set up in utils.py and may need some adjustments (TBC).


## Installing bplike

Inside the bplike directory, Simply do
```
$ pip install -e .
```


## Running chains with Cobaya

Simply run
```
$ cobaya-run bplike/run_scripts/ac_extended_act_only_bp.yml -f
```
and adapt the yml file acccording to what you would like to do.
