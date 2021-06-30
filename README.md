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

Data is located on NERSC at:

```
 /global/project/projectdirs/act/data/msyriac/bplike_data
 ```
You should download that directory and move it to bplike/data/.
And do the same with the repository actpolfull_dr4.01 that you will find on lambda at


 [https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.cfm](https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_get.cfm)



Data paths are set up in utils.py. Take a look at the paths there and make sure you have all the corresponding repositories, namely:

```
-/bplike_data/big_coadd_weights/200226/
-/bplike_data/coadd_data/
-/bplike_data/bpass/
-/actpolfull_dr4.01/data/data_act/
-/actpolfull_dr4.01/data/Fg/
```


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
[evaluate] log-likelihood = -537.626
[evaluate]    chi2_bplike.act100_act_only_TTTEEE = 567.175
[evaluate]    chi2_bplike.act15_act_only_TTTEEE = 508.078
[evaluate] Derived params:
[evaluate]    A_s = 2.12599e-09
[evaluate]    H0 = 67.0712
[evaluate]    sigma8 = 0.833653
[bplike.act100_act_only_ttteee] Average evaluation time for bplike.act100_act_only_TTTEEE: 2.19557 s  (1 evaluations)
[bplike.act15_act_only_ttteee] Average evaluation time for bplike.act15_act_only_TTTEEE: 1.6663 s  (1 evaluations)
[classy] Average evaluation time for classy: 1.7928 s  (1 evaluations)
```
To compute the Choi et al 20 best-fit model, without the bandpass integration, run:

```
$ cobaya-run run_scripts/act_extended_act_only.evaluate_bestfit.choietal20.yaml -f
```

which should return:

```
[evaluate]    chi2_bplike.act100_act_only_TTTEEE = 560.389
[evaluate]    chi2_bplike.act15_act_only_TTTEEE = 500.279
```

This reproduces exactly the best-fitting chi^2 reported in Choi el al (chi^2=1061).

(The dr4 fortran code returns 500.36 for deep and 560.25 for wide on this model.)


For these examples to run smoothly, you need to set your local path in the yaml file at the following entries:

```
output: /global/path/where/to/save/chains/act_extended_act_only_bp_classy_120621
```
