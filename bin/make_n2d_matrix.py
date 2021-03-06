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
pout,cout,sout = noise.get_save_paths(args.model,version,coadd,
                                      season=args.season,patch=args.patch,array=args.array,
                                      mkdir=True,overwrite=args.overwrite,mask_patch=mask_patch)
# Get data model
mask = sints.get_act_mr3_crosslinked_mask(mask_patch,
                                          version=args.mask_version,
                                          kind=args.mask_kind,
                                          season=args.season,array=args.array+"_f150",
                                          pad=args.mask_pad)
if args.debug: noise.plot(pout+"_mask",mask,grid=True)
dm = sints.models[args.model](region=mask,calibrated=args.calibrated)

assert not np.isnan(mask).any()
print(np.argwhere(np.isnan(mask)))

# Get a NoiseGen model
if args.extract_mask is not None:
    emask = sints.get_act_mr3_crosslinked_mask(mask_patch,version=args.extract_mask,kind=args.mask_kind,season=args.season,array=args.array+"_f150")
    eshape,ewcs = emask.shape,emask.wcs
else:
    emask = mask
ngen = noise.NoiseGen(version=version,model=args.model,extract_region=emask,ncache=1,verbose=True)

# Get arrays from array
print(dm.array_freqs[args.array])
splits = dm.get_splits(season=args.season,patch=args.patch,arrays=dm.array_freqs[args.array],srcfree=True)



assert splits.ndim==5
nsplits = splits.shape[1]
ivars = dm.get_splits_ivar(season=args.season,patch=args.patch,arrays=dm.array_freqs[args.array])
if args.debug: 
    noise.plot(pout+"_splits",splits)
    noise.plot(pout+"_ivars",ivars)


modlmap = splits.modlmap()
flatstring = "un" if args.do_only_filter_noise else ""
n2d_xflat = noise.get_n2d_data(splits,ivars,mask,coadd_estimator=coadd,
                               flattened=not(args.do_only_filter_noise),
                               plot_fname=pout+"_n2d_%sflat" % flatstring if args.debug else None,
                               dtype=dm.dtype)

print('A',np.argwhere(~np.isfinite(n2d_xflat)))

ncomps = n2d_xflat.shape[0]
if ncomps==1: npol = 1
else: npol = 3
mask_ell = args.rlmin - args.radial_fit_annulus
del splits

radial_pairs = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(0,3),(3,0)] if not(args.no_prewhiten) else []
if smooth:
    n2d_xflat_smoothed = noise.smooth_ps(n2d_xflat.copy(),dfact=dfact,
                                        radial_pairs=radial_pairs,
                                        plot_fname=pout+"_n2d_%sflat_smoothed" % flatstring if args.debug else None,
                                        radial_fit_annulus = args.radial_fit_annulus,
                                         radial_fit_lmin=args.rlmin,fill_lmax=args.lmax,log=not(args.delta_ell is None),delta_ell=args.delta_ell,nsplits=nsplits)
else:
    n2d_xflat_smoothed = n2d_xflat.copy()
del n2d_xflat

print('B',np.argwhere(~np.isfinite(n2d_xflat_smoothed)))

#n2d_xflat_smoothed[:,:,modlmap<mask_ell] = 0

fill_val = 0
if args.fill_const: fill_val = n2d_xflat_smoothed[:,:,np.logical_and(modlmap>args.fill_min,modlmap<(args.fill_min+args.radial_fit_annulus))].mean()
if args.fill_min is not None: n2d_xflat_smoothed[:,:,modlmap<args.fill_min] = fill_val
n2d_xflat_smoothed[:,:,modlmap<2] = 0
if args.lmax is not None: n2d_xflat_smoothed[:,:,modlmap>args.lmax] = 0

if args.no_off: n2d_xflat_smoothed = noise.null_off_diagonals(n2d_xflat_smoothed)

if args.do_only_filter_noise:
    ngen.save_filter_noise(n2d_xflat_smoothed,season=args.season,patch=args.patch,array=args.array,coadd=coadd,mask_patch=mask_patch)    
    sys.exit()

print('C',np.argwhere(~np.isfinite(n2d_xflat_smoothed)))

enmap.write_map(pout+'_n2d.fits',n2d_xflat_smoothed)

