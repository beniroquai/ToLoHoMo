#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:26:37 2022

@author: bene
"""
import h5py
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt
nip.setViewer("NIP_VIEW")

import tifffile as tif

mpathh = '/Users/bene/Dropbox/HoloTomo/2022-03-03_InlineHolo_488nm.tif - drift corrected.tif'
mstack = tif.imread(mpathh)




laserName = "635 Lasesr"
ledName = "White LED"

N_step_roundtrip = 53*300

#pixelsize
N_period = 20
P_period = 950 # pixels
d_reality = 25.4*1e-3/408 # dpi of huawei p20
print('pixelsize')
pixelsize = N_period*d_reality/P_period # 1.3106295149638802e-06 m
mWavelength = 488*1e-9
NA=.3
pixelsize_nyq = mWavelength/NA/2 # 0.8133333333333334e-06 => undersampled

k0 = 2*np.pi/(mWavelength)

PSFpara = nip.PSF_PARAMS()
PSFpara.wavelength = mWavelength
PSFpara.NA=NA
PSFpara.pixelsize = pixelsize



#%%
is_debug=True
N_subroi = 800
Nz = 20
dz=1*1e-3
images = []
for iimage in range (mstack.shape[0]):
    print(iimage)
    mimage = nip.image(np.sqrt(mstack[iimage ,:,:]))
    mimage = nip.extract(mimage, [N_subroi,N_subroi])
    mimage.pixelsize=(pixelsize, pixelsize)
    mpupil = nip.ft(mimage)
    


    propagated = nip.ift2d(nip.propagatePupil(mpupil, sizeZ=Nz, distZ=dz, psf_params=PSFpara, doDampPupil=True))
    
    if is_debug:
        for i in range(propagated.shape[0]):
            plt.imshow(np.abs(propagated[i,:,:]), cmap='gray'), plt.show()
    
    images.append(propagated[0,:,:])

tif.imwrite('tmpholo.tif',np.array(np.abs(images)))
hf = h5py.File('data.h5', 'w')
hf.create_dataset('dataset_1', data=np.array(images))
hf.close()