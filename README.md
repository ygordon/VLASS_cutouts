**get_VLASS_cutouts.py readme**

command line script for querying CADC for VLASS images and obtaining a cutout fits files centred on the coordinates provided. Where a cutout covers multiple VLASS images, the returned cutout is a mosaic including updated WCS and beam information in the header. Run as: 

    >python get_VLASS_cutouts.py target_list

where *target_list* is the filename of a table containing the right ascension and declination coordinates on which to centre the cutouts. This can be in any format recognised by astropy.table. The default assumption is that the RA and Decl. coordinates are given in decimal degrees. If this is not the case then the coordinate units will either need to be specified in the table metadata (if using e.g. fits or votable formats) or by using the *--posunits* argument (see below).

There are a number of optional arguments that can be called when running this script. These are [default values in square brackets]:\
*--epoch* [2], the VLASS epoch to query;\
*--size* ["2arcmin"], a string defining the angular size of cutout to obtain;\
*--racol* ["RA"], name of the RA column in the target_list file;\
*--decol* ["DEC"], name of the Decl. column in the target_list file;\
*--posunits* ["deg,deg"], units for ra,dec coordinates. If the target file contains metadata on the coordinate units then that information is used instead;\
*--outdir* ["."], directory to save cutouts to.


**Code Dependencies**\
The following python packages are required to run this code (version used in development):
* numpy (1.21.4)
* pyvo (1.2)
* wget (3.2)
* argparse (1.1)
* distutils (3.10.0)
* astropy (5.0)
* astroquery (0.4.5)
* reproject (0.8)
* radio_beam (0.3.3)
