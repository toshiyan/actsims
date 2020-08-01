"""
This script can be used to make a covsqrt and a few trial sims.
"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from pixell import enmap,enplot,fft
import numpy as np
import os,sys
from actsims import noise
from soapack import interfaces as sints
#from enlib import bench
import bench
from orphics import io,stats
import matplotlib.pyplot as plt
from tilec import covtools
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Make covsqrt, generate some test sims, make verification plots.')
parser.add_argument("version", type=str,help='A prefix for a unique version name')
parser.add_argument("model", type=str,help='Name of a datamodel specified in soapack.interfaces.')
parser.add_argument("--do-only-filter-noise", action='store_true',help='Do not do noise sim templates. Instead just do unflattened filter noise.')
parser.add_argument("--mask-version", type=str,  default="padded_v1",help='Mask version')
parser.add_argument("--mask-kind", type=str,  default="binary_apod",help='Mask kind')
parser.add_argument("--mask-patch", type=str,  default=None,help='Mask patch')
parser.add_argument("--mask-pad", type=int,  default=None,
                    help='Mask additional padding. No padding is applied to the extracted mask if any specified.')
parser.add_argument("--extract-mask", type=str,  default=None,
                    help='Make sims on the big mask but do all the analysis on an extract of this version.')
parser.add_argument("--covsqrt-kind", type=str,default="arrayops",help='Method for covsqrt.')
parser.add_argument("--season", type=str,help='Season')
parser.add_argument("--array", type=str,help='Array')
parser.add_argument("--patch", type=str,help='Patch')
parser.add_argument("--rlmin",     type=int,  default=300,help="Minimum ell.")
parser.add_argument("--fill-min",     type=int,  default=150,help="Minimum ell.")
parser.add_argument("--fill-const", action='store_true',help='Fill < fill-lmin with constant instead of zero. Constant is inferred from noise in annulus.')
parser.add_argument("-n", "--nsims",     type=int,  default=10,help="Number of sims.")
parser.add_argument("-r", "--radial-fit-annulus",     type=int,  default=20,help="Bin width for azimuthal averaging.")
parser.add_argument("-d", "--dfact",     type=int,  default=8,help="Downsample factor.")
parser.add_argument("--delta-ell",     type=int,  default=None,help="Downsample factor.")
parser.add_argument("-a", "--aminusc", action='store_true',help='Whether to use the auto minus cross estimator.')
parser.add_argument("--no-write", action='store_true',help='Do not write any FITS to disk.')
parser.add_argument("--calibrated", action='store_true',help='Apply default calibration factors to arrays.')
parser.add_argument("--no-off", action='store_true',help='Null the off-diagonals.')
parser.add_argument("--no-prewhiten", action='store_true',help='Do not prewhiten spectra before smoothing. Use this flag for Planck.')
parser.add_argument("--overwrite", action='store_true',help='Overwrite an existing version.')
parser.add_argument("--debug", action='store_true',help='Debug plots.')
parser.add_argument("--lmax",     type=int,  default=None,help="Maximum ell.")
args = parser.parse_args()
coadd = not(args.aminusc)
nsims = args.nsims
if args.mask_patch is None: mask_patch = args.patch
else: mask_patch = args.mask_patch
if args.dfact == 0: 
    smooth = False
else: 
    smooth = True
    dfact = (args.dfact,args.dfact)

# Make version tag
version = args.version
other_keys={'mask_version':args.mask_version}
for key in other_keys.keys():
    version += ("_"+key+"_"+str(other_keys[key]))
#####


# Get file name convention
pout,cout,sout = noise.get_save_paths(args.model,version,coadd,season=args.season,patch=args.patch,array=args.array,mkdir=True,overwrite=args.overwrite,mask_patch=mask_patch)

# Get data model
mask = sints.get_act_mr3_crosslinked_mask(mask_patch,version=args.mask_version,kind=args.mask_kind,season=args.season,array=args.array+"_f150",pad=args.mask_pad)
dm = sints.models[args.model](region=mask,calibrated=args.calibrated)

n2d = enmap.read_map(pout+'_n2d.fits')
covsq = enmap.multi_pow(n2d.copy(),0.5)
covsq[:,:,n2d.modlmap()<2] = 0
print('D',np.argwhere(~np.isfinite(covsq)))
covsq = enmap.enmap(covsq,n2d.wcs)
print('E',np.argwhere(~np.isfinite(covsq)))

#enmap.write_map(pout+'_covsqrt_tmp.fits',covsq)
#covsq = enmap.read_map(pout+'_covsqrt_tmp.fits')

#covsq = enmap.read_map(pout+'_covsqrt.fits')

#shape = np.shape(n2d[0,0,:,:])
#res = n2d*0.
#for li in range(shape[0]):
#    for lj in range(shape[1]):
#        E, V = np.linalg.eigh(n2d[:,:,li,lj])
#        res[:,:,li,lj] = V.dot(E[:,None]*V.T)

plt.imshow(np.log(covsq[0,0,:,:]+1e-30))
plt.savefig(pout+'_covsqrt.png')
plt.clf()
