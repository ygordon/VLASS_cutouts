###code to get VLASS cutout images from CADC

import numpy as np, pyvo as vo, wget, argparse, warnings, os
from distutils.util import strtobool
from astroquery.cadc import Cadc
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from spectral_cube import SpectralCube
from reproject import reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
import radio_beam as rb
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning) ###mutes warnings in printout

#########################################################################
#########################################################################
### run on command line as:
###    > python get_VLASS_cutouts.py target_filename
#########################################################################
#########################################################################

def iau_name(ra, dec, aprec=2, dp=5, prefix=''):
    ###truncated to dp so compatible with IAU standards
    ra, dec = np.array(ra), np.array(dec)
    
    cat = SkyCoord(ra=ra, dec=dec, unit='deg')
    
    astring = cat.ra.to_string(sep='', unit='hour', precision=dp, pad=True)
    dstring = cat.dec.to_string(sep='', precision=dp, pad=True, alwayssign=True)
    
    ###truncation index
    tinda = aprec+7
    tindd = tinda
    if aprec==1:
        tindd = tindd-1
    if aprec<1:
        tinda = tinda-1
    
    if prefix == '':
        sname = ['J' + astring[i][:tinda] + dstring[i][:tindd] for i in range(len(astring))]
    else:
        sname = [prefix + ' J' + astring[i][:tinda] + dstring[i][:tindd] for i in
                 range(len(astring))]
    
    return(sname)
    
    
def get_coordlist(data, racol='RA', decol='DEC', posu='deg,deg'):
    'extract a list of coordinates from data table'
    if data[racol].unit is None or data[decol].unit is None:
        coordlist = SkyCoord(ra=data[racol], dec=data[decol], unit=posu)
    else:
        coordlist = SkyCoord(ra=data[racol], dec=data[decol])
        
    return coordlist

    
def trim_slash(string):
    'remove forward slash from end of string, e.g. allows functions here to work if given directory name of format dir/ rather than dir'
    if string[-1] == '/':
        string = string[:-1]
    
    return string
    
    
def VLASS_image_query(position, size, imcodes=['ql', 'image'],
                      timeout=30):
    'query CADC for VLASS cutouts and return result'
    radius = size/2
    cadc = Cadc()
    cadc.TIMEOUT = timeout
    results = cadc.get_images(position, radius, collection='VLASS')
    
    ###create meta dict for image results
    outdict = {}
    for res in results:
        head = res[0].header
        hkeylist = list(head.keys())
        filnamecodes = [str(head[f]) for f in hkeylist if 'FILNAM' in f]
        if 'VLASS' in head['FILNAM01']:
            epoch = '.'.join([head['FILNAM01'], head['FILNAM02']])
            name = head['FILNAM05']
        else:
            epoch = '.'.join([head['FILNAM02'], head['FILNAM03']])
            name = head['FILNAM06']
        if all(f in filnamecodes for f in imcodes):
            outdict['_'.join([name, epoch])] = res
    
    return outdict


def make_vlass_coad(inputs):
    'make coadded VLASS image from inputs'
    
    ###load, reduce to 2D and extract necessary info
    cubes = [SpectralCube.read(i) for i in inputs]
    imdat = [(i.hdu.data[0], i.wcs.dropaxis(2)) for i in cubes]
    wcs_out, shape_out = find_optimal_celestial_wcs(imdat)
    
    ###find common beam and convolve images with
    beams = rb.Beams(beams = [rb.Beam.from_fits_header(i.header) for i in cubes])
    common = rb.commonbeam.common_manybeams_mve(beams)
    convolved = [i.convolve_to(common, convolve=convolve_fft) for i in cubes]
    indata = [(i.hdu.data[0], i.wcs.dropaxis(2)) for i in convolved]
    
    ###coad convolved images
    outdata, footprint = reproject_and_coadd(indata, wcs_out,
                                             shape_out=shape_out,
                                             reproject_function=reproject_exact)
    
    ###make output fits hdu and write to file
    hdu_out = fits.PrimaryHDU(outdata, header=wcs_out.to_header())
    hdu_out.header = common.attach_to_header(hdu_out.header) ##adds beam info
    ###add additional info to header (obs dates [median/range], comment on input images)?
    hdu_out = fits.HDUList(hdu_out)
    
    return hdu_out
    
    
def obtain_preferred_cutout(results_dict, pref_epoch=1):
    'output only the preferred epoch data'
    
    ###look for correct epoch in keys
    epkey = 'VLASS' + str(pref_epoch)
    keylist = list(results_dict.keys())
    right_epoch = [key for key in keylist if epkey in key]
    
    ###sift out data
    epoch_data = [results_dict[key] for key in keylist if key in right_epoch]
    
    ###if only one image returned then great, if more (e.g. e1.1, e1.2) stack/mosaic
    if len(epoch_data) == 1:
        hdu = epoch_data[0]
    else:
        hdu = make_vlass_coad(epoch_data)
    
    return hdu
    
    
def download_cutouts(position, size, outdir='.', epoch=1, filename=None,
                     overwrite_existing=False,
                     print_file_to_download=True,
                     timeout=30):
    'take input position and cutout size to save single cutout to file'
    ###can alter preferred epoch, will be a mosaic if multiple partial cutouts returned
    outdir = trim_slash(outdir)
    
    ###setup file name
    source_name = iau_name(ra=[position.ra.value],
                           dec=[position.dec.value],
                           aprec=2)[0]
    if filename is None:
        filename = '_'.join([source_name, '-'.join(['VLASS', str(epoch)])])
        filename = '.'.join([filename, 'fits'])
    
    ###check if file exists
    file_list = os.listdir(outdir)
    if overwrite_existing==False and filename in file_list:
        print('')
        print('WARNING: file ' + filename + ' already exists, not redownloading data')
        return
    if print_file_to_download==True:
        ###useful for identifying failed downloads
        print('downloading ' + filename)
    
    ###obtain data
    results = VLASS_image_query(position=position, size=size, timeout=timeout)
    
    ###extract required hdu
    hdu = obtain_preferred_cutout(results_dict=results, pref_epoch=epoch)
    
    ###add OBJECT to header
    hdu[0].header['OBJECT'] = ' '.join(['VLASS', source_name])
    
    ###save to file
    filename = '/'.join([outdir, filename])
    hdu.writeto(filename)
    
    return


def parse_args():
    "parse input args, i.e. target list, image, and rms file names"
    parser = argparse.ArgumentParser(description="download cutout fits images from VLASS")
    parser.add_argument("targets", help="file with list of sky coords to centre cutouts on")
    parser.add_argument("--epoch", action="store", type=int,
                        default=2, help="VLASS epoch to get images of")
    parser.add_argument("--size", action="store", type=str,
                        default="2arcmin", help="length of side of square cutout")
    parser.add_argument("--racol", action="store", type=str,
                        default="RA", help="column name with RA coordinates in target file")
    parser.add_argument("--decol", action="store", type=str,
                        default="DEC", help="column name with Dec. coordinates in target file")
    parser.add_argument("--posunits", action="store", type=str,
                        default="deg,deg", help="units to assume coordinates are in if not included in target table metadata")
    parser.add_argument("--mosaic", action="store", type=str,
                        default="True", help="mosaic if cutout straddles multiple images") ###need to add functionality to deactivate mosaicing
    parser.add_argument("--outdir", action="store", type=str,
                        default=".", help="directory to save files to")
    parser.add_argument("--timeout", action="store", type=int,
                        default=30, help="CADC query timeout limit")
    args = parser.parse_args()
    
    ###convert arg.model_psf to bool
    args.mosaic = bool(strtobool(args.mosaic))
    
    ###convert cutout size to Quantity
    args.size = u.Quantity(args.size)

    return args

#########################################################################
#########################################################################
###main


if __name__ == "__main__":
    ##parse command line arguments
    args = parse_args()

    ##load data
    targets = Table.read(args.targets)
    coordinates = get_coordlist(data=targets, racol=args.racol,
                                decol=args.decol, posu=args.posunits)
    for co in coordinates:
        download_cutouts(position=co, size=args.size, outdir=args.outdir,
                         epoch=args.epoch, timeout=args.timeout)
