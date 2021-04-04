# example cobaya-compliant ACT likelihood package;
# adapted from github.com/cobayasampler/example_external_likelihood

from setuptools import setup

setup(
    name="bplike",
    version="0.0",
    description="Multi frequency likelihood",
    zip_safe=False,
    packages=["bplike"], # add more things, e.g.:, "soliket.tests", "soliket.clusters", "soliket.ymap"],
    package_data={
        "bplike": [
                    "data/actpolfull_dr4.01/data/*",
                    "data/actpolfull_dr4.01/data/Fg/*",
                    "data/bplike_data/big_coadd_weights/200226/*",
                    "data/bplike_data/coadd_data/*",
                    "data/bplike_data/bpass/*",
                    "data/actpolfull_dr4.01/data/data_act/*txt",
                    "*.yaml",
                    "*.yml"
        ]
    },
    # install_requires=[
    #     "astropy",
    #     "scikit-learn",
    #     "cobaya",
    #     "sacc",
    #     "pyccl",
    #     "fgspectra @ git+https://github.com/simonsobs/fgspectra@master#egg=fgspectra",
    #     "mflike @ git+https://github.com/simonsobs/lat_mflike"
    # ],
    # test_suite="soliket.tests",
)
