import act_pylike

# Script to build and save coadd matrices
output_dir = "../data/coadd_data/coadds_20200305"

sbands = { 'TT':[('95','95'),('95','150'),('150','150')],
           'TE':[('95','95'),('95','150'),('150','95'),('150','150')],
           'EE':[('95','95'),('95','150'),('150','150')] }

for flux in ['15mJy','100mJy']:
    for spec in ['TT','TE','EE']:
        for bands in sbands[spec]:
            band1,band2 = bands
            print(flux,spec,band1,band2)
            act_pylike.save_coadd_matrix(spec,band1,band2,flux,output_dir)
