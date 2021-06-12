# bplike

This is a version of Erminia's multi-frequency likelihood
for the ACT DR4 release, but with the following changes:
- includes (optional) scale-dependent bandpass corrections,
- it is re-written in Python,
- It interfaces with [cobaya](https://github.com/CobayaSampler/cobaya).

It reproduces Erminia's results when run without the
bandpass corrections.

It currently has dependencies that include [soapack](https://github.com/simonsobs/soapack/tree/master/soapack) and [tilec](https://github.com/ACTCollaboration/tilec/tree/boris), which should ideally be removed.

bplike currently runs with either of the latest versions of [camb](https://github.com/cmbant/CAMB), [class](https://github.com/lesgourg/class_public), or [class_sz ](https://github.com/borisbolliet/class_sz).



## Data

located on NERSC:

```
 /global/project/projectdirs/act/data/msyriac/bplike_data
 ```

You should download that directory and move it to bplike/data/.
Data paths are set up in utils.py. Take a look at the paths there and make sure you have all these data (bandpass, beams, foregrounds, spectra, etc).


## Installing bplike

Inside the bplike directory, simply do
```
$ pip install -e .
```


## Running chains with Cobaya

A simple example is available here:
```
$ cobaya-run bplike/run_scripts/act_extended_act_only_bp.evaluate_bestfit.120621.yaml -f
```
You can then adapt this yaml file acccording to what you would like to do.

If set up properly, this should return the following:

```
[evaluate] log-likelihood = -537.313
[evaluate]    chi2_bplike.act100_act_only_TTTEEE = 567.182
[evaluate]    chi2_bplike.act15_act_only_TTTEEE = 507.443
[evaluate] Derived params:
[evaluate]    A_s = 2.12599e-09
[evaluate]    H0 = 67.0712
[evaluate]    sigma8 = 0.833653
[bplike.act100_act_only_ttteee] Average evaluation time for bplike.act100_act_only_TTTEEE: 2.21904 s  (1 evaluations)
[bplike.act15_act_only_ttteee] Average evaluation time for bplike.act15_act_only_TTTEEE: 1.66744 s  (1 evaluations)
[classy] Average evaluation time for classy: 1.77039 s  (1 evaluations)
```
For this to run smoothly, you need to set your local path in the yaml file at the following entries:

```
path: /global/path/to/class/repo/class_public
...
output: /global/path/where/to/save/chains/act_extended_act_only_bp_classy_120621
```
