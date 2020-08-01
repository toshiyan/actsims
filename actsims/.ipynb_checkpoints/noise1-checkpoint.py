import numpy as np
import os,sys
from pixell import enmap,enplot,fft as pfft
from soapack import interfaces as sints
from actsims import util
from enlib import bench
import warnings

from memory_profiler import profile

if 'fftw' not in pfft.engine: warnings.warn("No pyfftw found. Using much slower numpy fft engine.")

class NoiseGen(object):
    def __init__(self,version,qid=None,model="act_mr3",extract_region=None,extract_region_shape=None,extract_region_wcs=None,ncache=1,verbose=False):
        """
        version: The version identifier for the filename of covsqrts on disk
        model: The name of an implemented soapack datamodel
        extract_region: An optional map whose footprint on to which the sims are made
        extract_region_shape: Instead of passing a map for extract_region, one can pass its shape and wcs
        extract_region_wcs: Instead of passing a map for extract_region, one can pass its shape and wcs
        ncache: The number of 

        """
        self._qid = qid
        if self._qid is not None: print(self._qid,len(self._qid))
        self._version = version
        self._model = model
        self.ncache = ncache
        self._ccache = {}
        self._icache = {}
        self.dm = sints.models[model](region=extract_region)
        self.verbose = verbose

    def load_covsqrt(self,season=None,patch=None,array=None,coadd=True,mask_patch=None,get_geometry=False):
        pout,cout,sout = get_save_paths(self._model,self._version,coadd=coadd,
                                        season=season,patch=patch,array=array,
                                        overwrite=False,mask_patch=mask_patch)
        fpath = "%s_covsqrt.fits" % (cout)
        if get_geometry: return enmap.read_map_geometry(fpath)
        ikey = '_'.join( [ str(x) for x in [season,patch] ] )
        try:
            covsqrt = self._ccache[fpath]
            ivars = self._icache[ikey]
            if self.verbose: print("Loaded cached covsqrt and ivars.")
        except:
            if self.verbose: print("Couldn't find covsqrt and ivars in cache. Reading from disk...")
            ivars = enmap.enmap( [ self.dm.get_ivars( q ) for q in self._qid ] )
            covsqrt = enmap.read_map(fpath)
            print(fpath,covsqrt.shape)
            if len(self._ccache.keys())<self.ncache: 
                self._ccache[fpath] = covsqrt
                self._icache[ikey] = ivars
        return covsqrt,ivars
        
    def generate_sim(self,season=None,patch=None,array=None,seed=None,mask_patch=None,apply_ivar=True,ext_signal=0):
        covsqrt, ivars = self.load_covsqrt(season=season,patch=patch,array=array,mask_patch=mask_patch)
        sims, ivars = generate_noise_sim(covsqrt,ivars,seed=seed,dtype=self.dm.dtype)
        return sims,ivars


def get_save_paths(model,version,coadd,season=None,patch=None,array=None,mkdir=False,overwrite=False,mask_patch=None):
    paths = sints.dconfig['actsims']

    try: assert paths['plot_path'] is not None
    except: paths['plot_path'] = "./"
    assert paths['covsqrt_path'] is not None
    assert paths['trial_sim_path'] is not None

    # Prepare output dirs
    pdir = "%s/%s/" % (paths['plot_path'] ,version) 
    cdir = "%s/%s/" % (paths['covsqrt_path'] ,version)
    sdir = "%s/%s/" % (paths['trial_sim_path'] ,version)
    
    if mkdir:
        exists1 = util.mkdir(pdir)
        exists2 = util.mkdir(cdir)
        exists3 = util.mkdir(sdir)
        if any([exists1,exists2,exists3]): 
            if not(overwrite): raise IOError
            warnings.warn("Version directory already exists. Overwriting.")

    if model=='planck_hybrid': 
        assert season is None
        suff = '_'.join([model,patch,array,"coadd_est_"+str(coadd)])
    else:
        suff = '_'.join([model,season,patch,array,"coadd_est_"+str(coadd)])


    pout = pdir + suff
    cout = cdir + suff
    sout = sdir

    if mask_patch is not None:
        if mask_patch != patch:
            pout = pout+"_"+mask_patch
            cout = cout+"_"+mask_patch
            sout = sout+mask_patch+"_"

    return pout,cout,sout


@profile
def kmap_stack(kmap,wcs):
    return enmap.enmap(np.stack(kmap),wcs)


def generate_noise_sim(covsqrt,ivars,seed=None,dtype=None):
    """
    Supports only two cases
    1) nfreqs>=1,npol=3
    2) nfreqs=1,npol=1
    """
    if isinstance(seed,int): seed = (seed,)
    shape,wcs = covsqrt.shape,covsqrt.wcs
    Ny,Nx = shape[-2:]
    ncomps = covsqrt.shape[0]
    nfreqs = 1 if ncomps==1 else ncomps // 3
    if ncomps==1: npol = 1
    else: npol = 3
    wmaps = enmap.extract(ivars,shape[-2:],wcs)
    nsplits = wmaps.shape[1]

    if dtype is np.float32: ctype = np.complex64 
    elif dtype is np.float64: ctype = np.complex128 

    # Old way with loop
    kmap = []
    for i in range(nsplits):
        if seed is None:
            np.random.seed(None)
        else:
            np.random.seed(seed+(i,))
        rmap = enmap.rand_gauss_harm((ncomps, Ny, Nx),covsqrt.wcs).astype(ctype)
        kmap.append( enmap.map_mul(covsqrt, rmap) )
    del covsqrt, rmap
    del ivars
    kmap = kmap_stack(kmap,wcs)
    print(np.shape(kmap))
    #kmap = enmap.enmap(np.stack(kmap),wcs)
    outmaps = enmap.ifft(kmap[:,0:3,:,:], normalize="phys").real
    del kmap

    # Need to test this more ; it's only marginally faster and has different seed behaviour
    # covsqrt = icovsqrt 
    # np.random.seed(seed)
    # rmap = enmap.rand_gauss_harm((nsplits,ncomps,Ny, Nx),covsqrt.wcs)
    # kmap = enmap.samewcs(np.einsum("abyx,cbyx->cayx", covsqrt, rmap),rmap)
    # outmaps = enmap.ifft(kmap, normalize="phys").real

    # isivars = 1/np.sqrt(wmaps)
    with np.errstate(divide='ignore',invalid='ignore'): 
        isivars   = ((1./wmaps) - (1./wmaps.sum(axis=1)[:,None,...]))**0.5
    isivars[~np.isfinite(isivars)] = 0
    
    assert np.all(np.isfinite(outmaps))
    # Divide by hits
    for ifreq in range(nfreqs):
        outmaps[:,ifreq*npol:(ifreq+1)*npol,...] = outmaps[:,ifreq*npol:(ifreq+1)*npol,...] * isivars[ifreq,...] *np.sqrt(nsplits)

    retmaps = outmaps.reshape((nsplits,nfreqs,npol,Ny,Nx)).swapaxes(0,1)
    assert np.all(np.isfinite(retmaps))
    return retmaps,wmaps

    
