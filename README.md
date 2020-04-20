# bplike

This is a version of Erminia's multi-frequency likelihood
for the ACT DR4 release, but with the following changes:
- includes (optional) scale-dependent bandpass corrections 
- it is re-written in Python 
- it interfaces with Cobaya

It reproduces Erminia's results when run without the 
bandpass corrections.

It currently has dependencies that include soapack and tilec,
which should ideally be removed.