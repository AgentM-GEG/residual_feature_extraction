'''
This module contains the key functions that carryout the residual feature extraction on the input
GALFIT-based image cube.

Author: Kameswara Bharadwaj Mantha
email: km4n6@mail.umkc.edu
'''

############### loading the necessary modules ###########
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import os.path
import warnings
from skimage import measure
from astropy.visualization import (MinMaxInterval,PercentileInterval,ZScaleInterval,SqrtStretch,AsinhStretch,LogStretch,ImageNormalize)
import copy
#######################################


def writeCsvFile(fname, data, *args, **kwargs):
    '''
    :param fname: file name of the destination csv file.
    :param data: data to be written into the csv file
    :param args: additional args for the csv writer function
    :param kwargs: additional keyword arguments to the csv writer function
    :return: nothing. Writes the csv file at the destination location.
    '''
    import csv
    mycsv = csv.writer(open(fname, 'wb'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)
    return


def identify_objects(image_data,nsigma,min_area,deb_n_thresh,deb_cont,param_dict):
    '''
    This function performs source identification using the python-based module named SEP (Barbary et al., 2016).
    :param image_data: provide the image data as an numpy.ndarray.
    :param nsigma: significance above the computed background where sources are identified.
    :param min_area: minimum area of the contiguous regions to be identified as a source.
    :param deb_n_thresh: number of thresholds to be applied during deblending sources.
    :param deb_cont: deblend contrast ratio
    :param param_dict: parameter dictionary which contains user specified information about source extraction parameters.
    :return: object list identified in the image, segmentation map with each object labeled from 1, 2, 3 ... The catalog of objects is
    ordered as per the segmap number.
    '''

    import sep # the source extraction module.
    from matplotlib.patches import Ellipse # ellipse patches for visualziation purposes
    from astropy.convolution import Tophat2DKernel, Gaussian2DKernel, Box2DKernel #kernels used for convolution during source extraction
    from astropy.stats import gaussian_fwhm_to_sigma

    filter_kwarg = param_dict['sep_filter_kwarg'] # grab the filter keyword argument from the parameter file
    filter_size = float(param_dict['sep_filter_size']) # grab the filter size corresponding to the filter of choice.

    byte_swaped_data = image_data.byteswap().newbyteorder() # as suggested by the SEP documentation.

    global_bkg = sep.Background(byte_swaped_data) #calculating the global background

    bkg_subtracted = byte_swaped_data - global_bkg #background subtracted data

    if filter_kwarg.lower() not in ['tophat','gauss','boxcar']:#check if the filter keyword exists in list of supported filters.
        warnings.warn('The filter %s is not supported as of yet, defaulting to tophat of radius 5') #if doesn't exist, then warn the user
        source_kernel = Tophat2DKernel(5) #default to the Tophat kernel of radius 5 pixels
    elif filter_kwarg.lower() == 'tophat':
        source_kernel = Tophat2DKernel(filter_size)
    elif filter_kwarg.lower() == 'gauss':#for gaussian filter, the size provided should be a fwhm and it will be converted to a sigma internally.
        _gauss_sigma = gaussian_fwhm_to_sigma * filter_size
        source_kernel = Gaussian2DKernel(_gauss_sigma)#initiating a gaussian kernel
    elif filter_kwarg.lower() == 'boxcar':#boxcar kernel
        source_kernel = Box2DKernel(filter_size)# initiating a boxcar kernel of side length = filter_size

    ## Extract the objects and generate a segmentation map.
    objects, segmap = sep.extract(bkg_subtracted, nsigma, err=global_bkg.globalrms, minarea=min_area, deblend_nthresh=deb_n_thresh, deblend_cont=deb_cont,segmentation_map=True, filter_kernel=source_kernel.array)

    segmask = copy.deepcopy(segmap)#deep copy the segmap such that further manipulation doesn't affect the segmap
    skymask = invert_segmap(segmask) #invert the segmap into a binary mask, where the bkg sky is set to 1, and the sources to 0. This is the sky mask.
    
    ## save the segmentation map and the sky mask to respective fits files, with appropriate headers.
    fits.writeto(param_dict['fits_save_loc']+'/%s_%ssig_%sker_segmap.fits'%(os.path.basename(param_dict['gfit_outfile']).strip('.fits'),nsigma,param_dict['sep_filter_size']),segmap,header=get_original_image_header(param_dict['gfit_outfile']),overwrite=True)
    fits.writeto(param_dict['fits_save_loc']+'/%s_%ssig_%sker_sky_mask.fits'%(os.path.basename(param_dict['gfit_outfile']).strip('.fits'),nsigma,param_dict['sep_filter_size']),skymask,header=get_original_image_header(param_dict['gfit_outfile']),overwrite=True)


    if param_dict['sep_make_plots'].lower() == 'true':#if the user wishes to make and save plots during the source extraction process, then the following steps are taken.
        #initiate the figure and axes layout (2 column plot)
        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs
        #image visualization normalization using a 99 percentile of the pixel distribution and an hyperbolic arc sinosoid scaling (DS9 equivalent)
        img_norm = ImageNormalize(image_data, interval=PercentileInterval(99.),
                          stretch=AsinhStretch())

        ax1.imshow(image_data, origin='lower', cmap='gray', alpha=1,
                   norm=img_norm) #show the original image data on which source extraction is performed.
        for i in range(len(objects)):# for each object identified, lay down an ellipse as a DS9 marker.
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6 * objects['a'][i],
                        height=6 * objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax1.add_artist(e)
            ax1.text(objects['x'][i],objects['y'][i],'%s'%(get_segmap_value(segmap,(objects['x'][i],objects['y'][i]))),fontsize=10,color='black',ha='center',va='center')# at the center of each object indicate the segmap value.

        ax2.imshow(segmap, origin='lower', interpolation='nearest')# show the segmentation map.
        make_axis_labels_off(axs)# make the axis labels off for aesthetic reasons.
        plt.subplots_adjust(wspace=0)
        plt.savefig(param_dict['sep_plot_destin']+'/%s_source_extraction.png'%(os.path.basename(param_dict['gfit_outfile']).strip('.fits')), bbox_inches='tight') # save the plot as a png in the plotting folder.
        plt.close(fig)

    return objects, segmap # return the objects and segmentation map.

def get_segmap_value(segm,centroid):
    '''
    This function queries the segmentation value of an object using
    its x,y centroid.
    :param segm: segmentation image generated during source extraction
    :param centroid: centroid of the object whose seg value is required.
    :return: segmentation value
    '''
    y,x = centroid # as the array is flipped in terms of x and y, when querying the centroid value.
    return segm[int(x)][int(y)]# rounding the centroid to the nearest integer value as indices are not floats.

def fix_gfit_xy(x,y,naxis):
    '''
    This is to avoid some unforseen issues related to GALFIT-related cubes.
    Specially when some (often bright) sources outside the postage stamps are fit
    their x,y centroid values are off the image and quering their segmentation values
    will result an error. So, for such sources, we default them to the edges, this avoids code crashes.
    There is a cleaner way, to avoid such objects to be queried in the first place.
    :param x: xcentroid as given by GALFIT
    :param y: ycentroid as given by GALFIT
    :param naxis: naxis array [NAXIS1, NAXIS2] of the image
    :return: cleaned x,y values
    '''
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > naxis[0]:
        x = naxis[1] - 1
    if y > naxis[1]:
        y = naxis[0] - 1
    return x,y

def is_galfit(seg, gfit_info):
    '''
    This code figures out automatically if the objects identified in the source extraction process have been GALFITed.
    It is important to know if the sources need to be masked intelligently or simple segmap based masks.
    :param seg: segmentation map
    :param gfit_info: the header of the GALFIT model, which consists all the best-fit information.
    :return: an array of segmap values that have corresponding best-fit GALFIT models.
    '''
    how_many_gfit_sources = len(gfit_info['COMP_*']) #easy way to figure out how many best-fit single sersic models exist in the model header.
    required_array = []
    naxis = (gfit_info['NAXIS2'],gfit_info['NAXIS1'])
    for source in np.arange(1, how_many_gfit_sources + 1, 1):#for each source in the detected objects
        if len(gfit_info['%s_XC' % source].split()) !=1 and len(gfit_info['%s_YC' % source].split()) !=1:#to avoid stuff that is fixed and outside the stamps and/or sky.
            gfit_x, gfit_y = (float(gfit_info['%s_XC' % source].strip('[]').split()[0].strip('**')), float(gfit_info['%s_YC' % source].strip('[]').split()[0].strip('**')))
            gfit_x,gfit_y = fix_gfit_xy(gfit_x,gfit_y,naxis)#find the galfit centroid and fix if any issue.
            segm_of_gfit_gal = get_segmap_value(seg, (gfit_x, gfit_y))# get the segmentation value corresponding to that GALFIT centroid.
            required_array.append(segm_of_gfit_gal)#append to the empty list.

    return required_array#return the array of segmentation values that have corresponding best-fit models.


def invert_segmap(segmap):
    '''
    This function inverts an input segmentation map to make all the sources to 0 and the bkg sky to 1.
    :param segmap: Segmentation map.
    :return: inverted segmentation map
    '''
    incr_segmap = np.array(segmap) + 1 # increment the input segmentation map by 1.
    incr_segmap[incr_segmap >1] =0 # wherever the incremented map is larger than one, make it zero.
    return incr_segmap # now, all the sky pixels are ONE, and the rest are zeros.


def intelligent_masking(gal_centroid,objects,segmap,gal_pa,use_galfit_info,params):
    '''
    This code will generate masking regions for objects of interest, which includes the object at the center of the
    image, but not limited to it. The masks are elliptical in nature, centered at the centroid provided, with axis ratio
    from source extraction based values.
    :param gal_centroid: Centroid of the primary galaxy of interest
    :param objects: list of objects identified in the postage stamp during source detection process.
    :param segmap: segmentation map generated during the source extraction process.
    :param gal_pa: position angle of the primary galaxy of interest.
    :param use_galfit_info: use the galfit information or use the source extract information in the case of "forced sources"
    :param params: the parameter file dictionary
    :return: the inner mask of the primary galaxy and then followed by a list of other inner masks for forced objects that were galfitted.
    '''
    from matplotlib.patches import Ellipse
    galfit_model_header = get_model_header(params['gfit_outfile'])#get the model header that contains the best-fit model information
    central_gal_obj_number = get_segmap_value(segmap, gal_centroid) - 1 #get the object entry for the central object of interest
    if params['forced_segvalues_galfitted'] != '':#if the forced objects entry is not empty, the proceed
        other_forced_sources = list(np.array(params['forced_segvalues_galfitted'].split(',')).astype(int)) #make a list of objects that need to be force extracted features for.
    elif params['forced_segvalues_galfitted'] == '':#in case the forced values is empty, then, just declare an empty array.
        other_forced_sources = []
    if use_galfit_info: #if the other objects should use GALFIT centroid information during this intelligent masking...
        inner_mask_of_central = Ellipse(xy=gal_centroid, width=float(params['inner_mask_semi_maj']) * objects['a'][central_gal_obj_number],
                                        height=float(params['inner_mask_semi_min']) * objects['b'][central_gal_obj_number],
                                        angle=(gal_pa+90)*(np.pi/180.0), color='red',
                                        fill=False) # declare an elliptical mask object centered on the central galaxy's cetroid.
        other_masks = [] # declare an empty array in which other masks corresponding to forced sources will be populated.
        for forced_srcs in other_forced_sources: #for each forced sources in the list of forced sources...
            what_galfit_component = gfit_param_seg_mapping(galfit_model_header,segmap,forced_srcs) # figure out what GALFIT component does this object correspond to
            gfit_cen_x_other_source, gfit_cen_y_other_source = float(galfit_model_header['%s_XC'%what_galfit_component].split()[0].strip('[]')), float(galfit_model_header['%s_YC'%what_galfit_component].split()[0].strip('[]')) #figure out the centroid of this forced object
            another_mask = Ellipse(xy=(gfit_cen_x_other_source,gfit_cen_y_other_source), width=float(params['inner_mask_semi_maj']) * objects['a'][forced_srcs-1],
                                        height=float(params['inner_mask_semi_min']) * objects['b'][forced_srcs-1],
                                        angle=objects['theta'][forced_srcs-1]* 180. / np.pi, color='red',
                                        fill=False) # declare an inner elliptical mask for each forced source.
            other_masks.append(another_mask) # append this mask object to a list.
    else: # same as above, except that the forced sources don't use galfit centroid information.
        inner_mask_of_central = Ellipse(xy=gal_centroid, width=float(params['inner_mask_semi_maj']) * objects['a'][central_gal_obj_number],
                                        height=float(params['inner_mask_semi_min']) * objects['b'][central_gal_obj_number],
                                        angle=objects['theta'][central_gal_obj_number] * 180. / np.pi, color='red',
                                        fill=False)
        other_masks = []
        for forced_srcs in other_forced_sources:
            another_mask = Ellipse(xy=(objects['x'][forced_srcs-1],objects['y'][forced_srcs-1]), width=float(params['inner_mask_semi_maj']) * objects['a'][forced_srcs-1],
                                        height=float(params['inner_mask_semi_min']) * objects['b'][forced_srcs-1],
                                        angle=objects['theta'][forced_srcs-1]* 180. / np.pi, color='red',
                                        fill=False)
            other_masks.append(another_mask)

    return inner_mask_of_central, other_masks # return the inner mask of the primary galaxy of interest, followed by a list of mask objects for forced sources (if any provided by the user)

def make_intelligent_masked_data(inner_mask,centroid,main_array,feat_outer_rad,othermsks):
    '''
    This function creates a mask such that everthing inside an inner mask is set to zero,
    everything outside a physical circle is made to zero. It also accounts for all other masks
    including the forced objects that may have been galfitted.
    :param inner_mask: inner mask object created by intelligent masking function
    :param centroid: centroid of the primary object
    :param main_array: the image array
    :param feat_outer_rad: outer radius of the circle in pixels.
    :param othermsks: all other mask objects.
    :return: outer circle mask of a physical size, intelligent masked data
    '''
    from matplotlib.patches import Circle as circ#importing the circle object
    y_shape, x_shape = main_array.shape#getting the shape of the main image
    x_center, y_center = centroid#querying the centroid
    outer_patch = circ((x_center, y_center), feat_outer_rad, fill=False, color='red')#initiating the outer circular aperture
    mask = np.zeros(shape=main_array.shape) #intitiating a mask, but first filling it with zeroes.
    for i in range(x_shape):
        for j in range(y_shape):#basically iterating over each pixel
            # asking if the inner mask does not contain the point, within the outer circle and not fall in one of the other masks.
            if not(inner_mask.contains_point([i,j])) and outer_patch.contains_point([i,j]) and not any(o.contains_point([i,j]) for o in othermsks):
                mask[j][i] = 1 #if the condition is met, then assign a value of 1.

    return outer_patch, mask

def make_residual_segmask(segmap,centroid,forced_obj_number):
    '''
    This function makes a mask such that the regions available for residual feature extraction
    are made 1, and the rest to zeros. In this case all the sources except for the primary object's
    segmentation region and also the seg regions of the forced objects are unlocked.
    :param segmap: semgentation map generated during the feature extraction process.
    :param centroid: centroid of the primary object
    :param forced_obj_number: a list of objects numbers that are forced to be unlocked.
    :return: residual segmask
    '''
    incremented_segmask = np.array(segmap) + 1 #increment the segmentation map
    if forced_obj_number.split(',') !=['']:#making sure that it is not empty
        forced_objs = list((np.array(forced_obj_number.split(',')).astype(int))+1)#forced objects as integer type
    else:
        forced_objs = [] #else the forced list is empty.
    avoid_segmap_value = list(forced_objs)+[get_segmap_value(segmap,centroid)+1] # make a new list of objects whose seg map values need to be ignored.
    incremented_segmask[np.where((incremented_segmask >1) & ~np.isin(incremented_segmask, avoid_segmap_value))] = 0 # set all such regions to 0
    incremented_segmask[incremented_segmask>1] = 1 # all else to ones..
    return incremented_segmask #return the resultant map.

def sigma_mask(data,clip_lim):
    '''
    This function basically returns an image such that the pixels
    satisfy a flux-wise clipping limit, i.e., pix brighter than a value.
    :param data: input image
    :param clip_lim: what is the value above which the pixels need to be chosen
    :return: a new image with only pixels above the provided clipping value.
    '''
    copy_data = np.zeros(shape=data.shape) #make a 2d array with zeros, matching the actualy image dimensions.
    x_shape,y_shape = data.shape # unpack the shape.
    for x in range(0,x_shape,1):
        for y in range(0,y_shape,1):#basically iterating over each pixel.
            if data[x][y] >=clip_lim:#if the pixel value is above the limit
                copy_data[x][y] =data[x][y] #assign the corresponding pixel value of the value in the data.
            else:
                pass#do nothing
    return copy_data#return the new image with pixels whose values are larger than the limit.

def border_mask(image_data,border_value):
    '''
    Sometimes, certain things are present just to annoy you. One of such things are image edges.
    Here is short function that makes a mask such that the border values are assigned zero and remaining
    to one.
    :param image_data: image array
    :param border_value: what is the value of the border when you inspected say with DS9 or loaded using Python.
    :return: a mask such that the border pixels are made zero and ones wherever the image exists
    '''
    data_copy = np.zeros(shape=image_data.shape)#make a new array of zeros
    data_copy[image_data != border_value] = 1#if the pixels don't equal to the border value, make them ones.
    return data_copy#return the border mask


def gfit_param_seg_mapping(resid_info,seg_map,seg_value_of_interest):
    '''
    One of the challenges in running my own source extraction and using galfit results from somewhere else.
    It is key to map what extracted object corresponds what component in the galfit best-fit model information.
    This function maps an object's segmentation map value to its corresponding GALFIT component (assuming
    single sersic modeling).
    :param resid_info: the header information of the GALFIT cube, which contains the GALFIT best-fit information.
    :param seg_map: segmentation map of the image, generated during source extraction
    :param seg_value_of_interest: segmap value of interest, whose GALFIT component one wishes to know.
    :return: component number corresponding to the segmap value of interest.
    '''
    comp_in_gfit = 1#initiating a dummy value
    naxis = (resid_info['NAXIS2'],resid_info['NAXIS1']) #getting to know the naxis
    for each in np.arange(len(resid_info['COMP_*']))+1:#for each object in the range of available components in the header
        if resid_info['COMP_%s'%(each)] !='sky':#make sure the component is not a background sky.
            x_val, y_val = (float(resid_info['%s_XC'%each].split()[0].strip('**').strip('[]')),float(resid_info['%s_YC'%each].split()[0].strip('**').strip('[]')))
            x_val,y_val = fix_gfit_xy(x_val,y_val,naxis) # get the x,y centroid of that object and fix it if the object seems off the object.
            if get_segmap_value(seg_map,(x_val,y_val)) == seg_value_of_interest:#get the segmap value of that centroid, if that equals the segvalue of interest
                comp_in_gfit = each#recording the component number to that component.
    return comp_in_gfit #return the matched component number.




def get_orig_and_feature(file_name, axes,redshift,nsigma,ann_mode,kernel_size,parameters):
    '''
    The primary crux of the whole pipeline lies in this function.
    :param file_name: name of the galfit output file to process through the feature extraction
    :param axes: list of external plotting axes that can be used for several internal visualization process.
    :param redshift: redshift of the galaxy that you are running feature extraction
    :param nsigma: the significance above which the extracted feature's pixels need to satisfy.
    :param ann_mode: what annulus mode, leave this to "intel" for intelligent masking scheme, which works the best.
    :param kernel_size: size of the kernel used to smooth the residual image.
    :param parameters: parameter file dictionary, where certain internal functionalities need.
    :return: org_data, feature, MAG, zero_point, orig_PS, (XC,YC), ann_mode, gfit.mean.value,abs(gfit.stddev.value), nsigma, kernel_size,PHOTFNU, r_for_im_scale
    1. original image data; 2. the extracted feature map; 3. magnitude in AB of the best-fit model; 4. magnitude zero point
    5. pixel scale in arcseconds; 6. GALFIT X and Y centroid of the primary galaxy of interest; 7. what annulus mode was used;
    8. mean of the best-fit gaussian to the background sky; 9. Standard deviation (1 sigma) of the best-fit gaussian to the bkg sky;
    9. what significance was used above which the extracted feature pixels are chosen; 10. kernel size used; 11. calibration factor
    to convert the units of the image (e/s) into Janskies (Jy); 12. the length of the 5 kpc scale bar for plotting purposes.
    '''
    from astropy.modeling import models, fitting #load modules
    from astropy.cosmology import Planck15 #load cosmology module.



    axis1, axis2, axis3,axis4 = axes # unpack axes.

    cube = fits.open(file_name, memmap=True) #open the GALFIT cube.

    org_data = cube[1].data #original data used for fitting is in 1st HDU
    model_data = cube[2].data # best-fit model image data, i don't use this.
    res_data = cube[3].data # residual image data

    orig_header = cube[1].header #original image header

    galfit_info = cube[2].header #model image header, which contains the best-fit information
    cube.close() # close the cube file to prevent further i/o operations on it.

    NAXIS = int(orig_header['NAXIS2']), int(orig_header['NAXIS1']) # get the dimensions of the image.

    #Perform the source identification on the original image data. Use the parameter file to pass user specified
    # source extraction related parameters. This generates a list of objects and the segmentation map.
    detected_sources, segmentation_map = identify_objects(org_data, float(parameters['sep_sigma']), int(parameters['sep_min_area']),
                                                          int(parameters['sep_dblend_nthresh']), float(parameters['sep_dblend_cont'])
                                                          ,parameters)
    #by construction, the center of the image should have the galaxy of interest.
    # I query the segmentation map value corresponding to the center pixel of the image.
    seg_value_of_interest = get_segmap_value(segmentation_map,(NAXIS[1]/2.0,NAXIS[0]/2.0))
    # The component number in the galfit best-fit model header information corresponding to the primary
    # galaxy of interest might change from person-to-person. van der Wel+12 made sure that this object be always
    #the first object to be fit in the GALFIT parameter file. However, if i decide to run my own GALFIT-based pipelines
    # the object might be a 3rd object or 2nd object depending on the objects identified in the image. Therefore,
    # i map the primary object of interest to its corresponding GALFIT component in the following step.
    object_in_galfit = gfit_param_seg_mapping(galfit_info,segmentation_map,seg_value_of_interest)

    #get the X and Y GALFIT centroid of the best-fit model information
    XC, YC = float(galfit_info['%s_XC' % object_in_galfit].strip('[]').split()[0]), float(galfit_info['%s_YC' % object_in_galfit].strip('[]').split()[0])

    PA = float(galfit_info['%s_PA'%object_in_galfit].strip('[]').split()[0]) # get position angle


    AR = float(galfit_info['%s_AR'%object_in_galfit].strip('[]').split()[0]) # axis ratio

    try:
        orig_PS = orig_header['CD2_2'] * 3600.0 #(pixel size in arcseconds.)
    except KeyError:
        try:
            orig_PS = orig_header['PC2_2'] * 3600.0 # in case the above thing gives an error, try the other key word
        except KeyError: 
            try:
                orig_PS = abs(orig_header['CDELT1']) * 3600.0 # in case the above thing gives an error, try the other key word
            except KeyError:
                #if the above step also failed.. then your images might be missing an important pixel scale keyword.
                print 'Something bad has happend and the header corresponding to pixel scale not found'
    zero_point = galfit_info['MAGZPT'] #magnitude zeropoint
    MAG = float(galfit_info['%s_MAG'%object_in_galfit].strip('[]').split()[0]) # apparent magnitude of the object using the best-fit model
    
    try:
        PHOTFNU = float(orig_header['PHOTFNU']) # query the calibration factor.
    except KeyError: #in case you are using SDSS images, then it expects a different key word
        try:
            PHOTFNU = float(orig_header['NMGY']) # try the other key word, which is nano-maggies per count
        except KeyError:
            PHOTFNU = 1.5E-7 # if all fails, then assuming that the image units are in e/s, assumes a 1.5E-7 value sort of the approximate value for HST F160W CANDELS images
            print 'Assuming a calibration of 1.5E-7 Jy sec/electron conversion, if the image units are in electrons/sec'

    arcsec_per_kpc = Planck15.arcsec_per_kpc_proper(redshift).value #conversion factor of arcseconds per kilo-parsec


    rmax_mask = (arcsec_per_kpc * float(parameters['outer_phy_mask_size'])) / orig_PS # radius of the outer circle in pixels
    r_for_im_scale = (arcsec_per_kpc * 5) / orig_PS #size of the 5 kpc bar in pixels, later for plotting.

    conv_res_data = smooth_tidal_feature(res_data,kernel_size) # smooth the residual data with a boxcar kernel of the provided size.

    # create the mask of the core region and other masks based on the provided feature extraction parameter file choices.
    inner_patch, all_other_masks = intelligent_masking((XC, YC), detected_sources,segmentation_map,PA,use_galfit_info=True,params=parameters)

    #using the inner mask and other masks produced in the above step, make the masked data array.
    outer_mask, masked_data = make_intelligent_masked_data(inner_patch,(XC, YC), conv_res_data, rmax_mask,all_other_masks)

    #generate the background sky mask.
    noise_data_mask = invert_segmap(segmentation_map)

    # Here is an important step to figure out which regions of the image need to be forcibly unlocked.
    #in some cases, you may want to extract the features around a nearby galaxy that was also modelled by GALFIT
    # in some cases, you may come across some cases where the extended tidal features may be identified as a source
    # during source extraction, which you want to avoid as such sources will be automatically masked.
    # in the following steps, i am checking the four possibilities and creating an array of seg values to be
    # force unlocked.
    if parameters['forced_segvalues_galfitted'] == '' and parameters['forced_segvalues_not_galfitted'] !='':
        all_forced_segvalues =  parameters['forced_segvalues_not_galfitted'].split(',')
    elif parameters['forced_segvalues_galfitted'] != '' and parameters['forced_segvalues_not_galfitted'] =='':
        all_forced_segvalues = parameters['forced_segvalues_galfitted'].split(',')
    elif parameters['forced_segvalues_galfitted'] != '' and parameters['forced_segvalues_not_galfitted'] !='':
        all_forced_segvalues = parameters['forced_segvalues_galfitted'].split(',') + parameters['forced_segvalues_not_galfitted'].split(',')
    else:
        all_forced_segvalues = ['']
        # parameters['forced_segvalues_galfitted'].split(',')
    seg_values_to_unlock = ','.join(all_forced_segvalues) # join all the contents of the list with a ',' into a string.
    #generate a residual extraction mask in the following step.
    resid_segmask = make_residual_segmask(segmentation_map,(XC,YC),forced_obj_number=seg_values_to_unlock)

    masked_data = masked_data * resid_segmask # to combine the binary masks generated during annulus masking and segmasking
    # i multipy the masked_data which is the binary intelligent masks by the segmap based mask.

    mask_border = border_mask(org_data,0) # generating a border mask, in case if the galaxy is on the image edge.

    image_norm = ImageNormalize(org_data, interval=PercentileInterval(99.),
                          stretch=AsinhStretch()) # a ds9 stretch and scaling.
    axis1.imshow(org_data, origin='lower', cmap='gray', alpha=1,
                 norm=image_norm) # visualize the original image of the galaxy in axis 1
    axis2.imshow(res_data, origin='lower', cmap='gray', alpha=1,
                 norm=image_norm)
    axis3.imshow(org_data, origin='lower', cmap='gray', alpha=1,
                 norm=image_norm) # visualize the residual on axes 2 and 3


    masked_res = masked_data*conv_res_data # now its time to generated a masked residual by multiplying the mask with data.
    noise_ann = noise_data_mask *conv_res_data # similarly, i generate a sky-only region by multiplying the noise mask with smoothed residual

    # save the residual extraction mask in the user-specified location, with the header of the original image preserved.
    fits.writeto(parameters['fits_save_loc']+'/%s_%ssig_%sker_resid_extr_mask.fits'%(os.path.basename(parameters['gfit_outfile']).strip('.fits'),nsigma,kernel_size),resid_segmask * masked_data,header=orig_header,overwrite=True)

    # make a normalized histogram of the sky pixel regions, accounting for possible image edge.
    histo = np.histogram(np.array([i for i in (noise_ann*mask_border).flat if i!=0]), bins=100,density=True)


    bins = histo[1] # get the automatic bin edges.
    bin_centers = 0.5 * (bins[1:] + bins[:-1]) # get the bin centers corresponding to the bin edges.
    guess_mean = np.nanmean(np.array([i for i in (noise_ann*mask_border).flat if i!=0])) # guess the mean of the distribution.
    guess_std = np.nanstd(np.array([i for i in (noise_ann*mask_border).flat if i!=0])) # guess the standard deviation of the distribution.
    print 'guess_mean = %s'%guess_mean
    print 'guess_std = %s'%guess_std
    g_init = models.Gaussian1D(amplitude=np.max(histo[0]), mean=guess_mean, stddev=guess_std) # a gaussian model initiated with the guesses.
    fit_g = fitting.LevMarLSQFitter() # initiating a LMSQ fitter.
    # in case there is leaking flux from outer isophotes of bright source in image (despite being very leniant during source extraction)
    # we fit the gaussing to 10 bins after the peak of the histogram to avoid possible issues (if any) from systematically brigher pixels.
    gfit = fit_g(g_init, bin_centers[:np.argmax(histo[0])+10], histo[0][:np.argmax(histo[0])+10])

    #computing the difference in the difference in peaks of best-fit mean and of the distribution.
    # this is to ensure that the 1 sigma is not affected significantly by the difference in peaks.
    diff_in_peaks = abs(gfit.mean.value) - np.median(np.array([i for i in (masked_res * mask_border).flat]))
    feature = sigma_mask(masked_res, nsigma * abs(gfit.stddev.value) - diff_in_peaks)
    # identify the corresponding pixels in the masked, smoothed residual image, whose values are above
    # the user specified value of sigma, accounting for the possible difference in the peaks (usually very small!)

    # return all the important stuff..
    return org_data, feature, MAG, zero_point, orig_PS, (XC,YC), ann_mode, gfit.mean.value,abs(gfit.stddev.value), nsigma, kernel_size,PHOTFNU, r_for_im_scale



# def get_x_y_pix_feature(feature):
#     x_shape, y_shape = feature.shape
#     positions = []
#     for x in range(0, x_shape, 1):
#         for y in range(0, y_shape, 1):
#             if feature[x][y]!=0:
#                 positions.append([x,y])
#     return positions


def get_feature_mask(feat):
    '''
    Converts the extracted feature image into a binary mask
    :param feat: extracted feature image
    :return: a binary feature mask.
    '''
    null_array = np.zeros(shape=feat.shape)
    x_shape, y_shape = feat.shape
    for i in range(x_shape):
        for j in range(y_shape):
            if feat[i][j]!=0:
                null_array[i][j] =1
    return null_array

def area_based_spatial_localization(mask,feature,area_lim):
    '''
    This is an important function that identifies contiguous pixel regions and then
    uses their area attributes to isolate those regions that have areas above a user-provided
    area threshold.
    :param mask: a binary mask where the pixels corresponding to extracted features are ONES. All else are zeros.
    :param feature: the feature where the pixel values are those of the residual image.
    :param area_lim: what is the area threshold above which the contiguous regions need to be considered significant.
    :return: intensity image corresponding to the contiguous regions, a list of pixel positions for each contiguous region,
    list of areas of the contiguous pixel regions, list of unique values assigned to each contiguous region.
    '''
    from skimage import measure # I am glad that someone made this package
    blobs_labels = measure.label(mask, background=0) # where the labeling of the contiguous regions happens.

    blob_props = measure.regionprops(blobs_labels, intensity_image=feature) # for the region labels, the
    # properties of each contiguous region is quantified. The intensity image is where the pixels are queried
    # for each contiguous regions. This is in turn used to quantify the flux per contiguous region and also
    # their areas.

    areas = [i.area for i in blob_props if i.area>area_lim] # make a list of ares of contiguous regions above the area threshold
    blob_img_info = []
    blob_positions = []
    blob_areas = []
    for each in blob_props:#looping through the contiguous region properties
        if each.area in areas: # looping on each to see if the area is the array of areas. PS: this is a stupid way of doing this, i guess i was tired when i wrote this.
            blob_img_info.append(each.intensity_image)
            blob_positions.append(each.coords)
            blob_areas.append(each.area)

    return blob_img_info, blob_positions, blob_areas,blobs_labels # return important information.



def calculate_sb(localized_feat,areas,pix_scale,sk_mean,sk_bkg,pht_fnu,exp_time_gain):
    '''
    This function calculates the surface brightness of the extracted features.
    :param localized_feat: localized features, meaning only those regions that are above a certain area threshold.
    :param areas: list of the contiguous region areas
    :param pix_scale: the pixel scale in arcseconds
    :param sk_mean: mean of the best-fit gaussian to the background sky
    :param sk_bkg: the 1 sigma background value (std of the best-fit gaussian to the background sky)
    :param pht_fnu: the calibration factor to convert the image units of e/s to Jy. Pht_nu has units Jy * sec/electron
    :param exp_time_gain: exposure time.
    :return: surface brightness, error on the measured surface brightness, and a whole bunch of step-wise information.
    '''
    flux =[]
    for each in localized_feat:# for each contiguous region, append the cumulative flux to make an array of fluxes
        flux.append(sum(each.flat))

    print 'PHOTNU = %s Jy*sec/electrons'%pht_fnu
    obs_flux = sum(flux) * pht_fnu # flux in units of Janskies
    print 'flux = %s electrons'%sum(flux)
    print 'flux = %s Jy' % obs_flux


    obs_area = sum(areas) # total area in pixels
    print 'observed area = %s pixels'%obs_area
    print 'sky mean = %s electrons'%sk_mean
    print 'sky_std = %s electrons'%sk_bkg

    total_sky_mean = obs_area * sk_mean * pht_fnu #total contribution of sky mean flux to the total area
    print 'total sky mean = %s Jy'%total_sky_mean

    print 'Median exposure time = %s sec' %exp_time_gain
    delta_f = np.sqrt((obs_area*(sk_bkg))**2 + (sum(flux))/exp_time_gain)
    delta_f = delta_f * pht_fnu # flux error
    print 'delta f = %s Jy'%delta_f

    delta_f_over_f = delta_f/obs_flux

    magnitude = -2.5*np.log10(obs_flux - total_sky_mean) + 8.90 # standard AB mag
    print 'magnitude = %s AB'%magnitude
    sb = magnitude+2.5*np.log10(obs_area*pix_scale*pix_scale) # surface brightness = Flux/Area, when converted to mag/arcsec^2 = m +2.5log(A)
    print 'area = %s arcsec^(2)'%(obs_area*pix_scale*pix_scale)
    print 'surface brightness = %s AB arcsec^(-2)'%sb
    d_sb = 1.09 * delta_f_over_f
    information_array = [sum(flux),obs_flux, pht_fnu, obs_area, sk_mean, sk_bkg, total_sky_mean, delta_f, magnitude, obs_area*pix_scale*pix_scale, sb, d_sb]
    return sb, d_sb, information_array # return important information.

def make_TF_plotting_array(pos,TF):
    '''
    This function makes a plotting friendly image based on the positions of
    the contiguous regions of the features.
    :param pos: a list contiguous regions, where each contiguous region is a list of pixel coordinates.
    :param TF: the residual image, from which the pixel value is taken.
    :return: a plotting array, where the residual feature has finite pixels values and the rest are nans.
    '''
    plotting_array = np.zeros(TF.shape) # initiate an array of zeroes.
    plotting_array[plotting_array==0] = None # make all of them into Nones or nans.. i feel like there must be a better way to do this.
    for each_blob in pos: # for each contiguous region
        for positions in each_blob: # for each position of each contiguous region
            x,y = positions # grab the x and y pos
            plotting_array[x][y] = TF[x][y] # make the corresponding pixel in plotting array the same from residual image.

    return plotting_array # return an image for plotting.

def smooth_tidal_feature(feat,box_size):
    '''
    This function smooths the given image with a 2D box car filter of a user defined kernel size.
    :param feat: an input image one wants to smooth with a 2D box filter.
    :param box_size: what is the side length of the 2D box filter kernel?
    :return: Boxcar smoothed input image.
    '''
    from astropy.convolution import Box2DKernel, convolve # import necessary module
    kernel = Box2DKernel(box_size) # initiate the kernel
    convolved_feature = convolve(feat,kernel) # convolve the image with the kernel

    return convolved_feature # return the smoothed image.

def gen_binary_map_for_final_feat(coords, shape):
    '''
    A function to generate a binary image of the feature. This function is used during contour plotting.
    :param coords: a list of lists, where each list is a contiguous region and the corresponding region's coordinates
    :param shape: shape of the image
    :return: a binary mask image, where the feature regions are set to ONE, and vice-versa
    '''
    zero_image = np.zeros(shape=shape)
    for each_blob in coords:
        for position in each_blob:
            x, y = position
            zero_image[x][y] = 1
    return zero_image

def draw_feat_contours(localized_feat_coords, shape, axis,style_dict):
    '''
    A function that draws contours around the extracted features.
    :param localized_feat_coords: feature coordinates
    :param shape: shape of the image
    :param axis: axis onto which the contours need to plotted
    :param style_dict: a list of [line_style_of_inner_contour, line_style_of_outer_contour, line width of inner contour, line width of outer contour]
    :return: nothing
    '''
    ls_inner_cont, ls_outer_cont, lw_inner_cont,lw_outer_cont = style_dict
    binary_image = gen_binary_map_for_final_feat(localized_feat_coords, shape)

    contours_feature = measure.find_contours(binary_image, 0)
    for n, contours_feature in enumerate(contours_feature):
        axis.plot(contours_feature[:, 1], contours_feature[:, 0], linewidth=lw_inner_cont, color='white', linestyle=ls_inner_cont)
        axis.plot(contours_feature[:, 1], contours_feature[:, 0], linewidth=lw_outer_cont, color='black', linestyle=ls_outer_cont)
    return

def zoom_in_image(axis,shape,centers,zoom_factor):
    '''
    In case the primary galaxy is relative zoomed out in terms of visualization,
    this function will zoom in on the primary object.
    :param axis: what is the axis on which this object is being plotted.
    :param shape: dimensions of the image that is being plotted.
    :param centers: x and y centroids of the primary object
    :param zoom_factor: what is the zoom factor. 0.75 means zoom by 75% of the current size.
    :return: nothing
    '''
    ysh, xsh = shape # get the image shape
    x_cen, y_cen = centers # get the centroids
    axis.set_xlim([x_cen - (zoom_factor/2.0) * xsh, x_cen + (zoom_factor/2.0) * xsh]) # basically setting x and y limits
    axis.set_ylim([y_cen - (zoom_factor/2.0) * ysh, y_cen + (zoom_factor/2.0) * ysh])
    return

def draw_kpc_scale(r_line,axis,percent):
    '''
    Draws a line that corresponds to a physical scale
    :param r_line: length of the line.
    :param axis: axis on which the line needs to be drawn.
    :param percent: percent from the top of the image where the line needs to plotted.
    :return: shape of xaxis, y position where the line is placed.
    '''
    y_shape, x_shape = (int(axis.get_ylim()[1] - axis.get_ylim()[0]), int(axis.get_xlim()[1] - axis.get_xlim()[0]))
    #get the x and y shape.

    linex = np.arange(x_shape - r_line / 2.0, x_shape + r_line / 2.0,
                      0.01) # the line that will be drawn.

    axis.plot(linex,np.ones_like(linex) * percent*int(axis.get_ylim()[1]), marker=None, linestyle='-', linewidth=4,
            color='white') # plot the line

    return (x_shape, 0.98*int(axis.get_ylim()[1])) # return the shape of xaxis, y position where the line is placed


def make_axis_labels_off(axes):
    '''
    This function switches off the axis labels for a list of input axes.
    :param axes: a list of axes
    :return: nothing
    '''
    for each in axes.flat:#for each axis...
        each.xaxis.set_visible(False) #switch off the xaxis labels
        each.yaxis.set_visible(False) # switch off the yaxis labels
    return


def parse_param_file(file_name):
    '''
    The primary function that parses the feature extraction file
    :param file_name: name of the parameter file
    :return: dictionary that contains the parameter key words and corresponding entries.
    '''
    with open(file_name,mode='r') as param_file:# with the file open
        content = param_file.readlines() # read the lines
    param_dict = dict()#initiate an empty dictionary
    for each in content:# for each line in content
        if each !='\n' and not(each.startswith('#')):# if the line doesn't start with a comment
            key, value = each.strip().split('=')# split the entry on a "=" sign
            param_dict[key.strip(' ')] = value.strip(' ')#stripping any spaces
    return param_dict#return the parameter dictionary

def default_param_dict():
    '''
    This function provides a default parameter dictionary
    :return: a default parameter file dictionary
    '''
    default_dict = dict()

    default_dict['sigma'] = 2
    default_dict['boxcar_smooth_size'] =3
    default_dict['forced_segvalues_galfitted'] = ''
    default_dict['forced_segvalues_not_galfitted'] = ''
    default_dict['inner_mask_semi_maj'] = 1.5
    default_dict['inner_mas_semi_min'] = 1.5
    default_dict['outer_phy_mask_size'] = 30.0

    default_dict['run_MC'] = 'True'
    default_dict['plot_destin'] = os.getcwd()+'/'#in case the directory is not given, then the current working directory is chosen.
    default_dict['SB_info_path'] = os.getcwd()+'/'

    default_dict['sep_sigma'] = 0.75
    default_dict['sep_min_area'] = 7
    default_dict['sep_dblend_nthresh'] = 32
    default_dict['sep_dblend_cont'] = 0.001
    default_dict['sep_filter_kwarg'] = 'tophat'
    default_dict['sep_plot_destin'] = os.getcwd()+'/'
    default_dict['sep_make_plots'] = 'True'
    default_dict['fits_save_loc'] = os.getcwd()+'/'

    default_dict['generate_voronoi_maps'] = 'False'
    default_dict['vor_plot_save_loc'] = os.getcwd() + '/'
    default_dict['vor_fits_save_loc'] = os.getcwd() + '/'
    default_dict['vor_data_save_loc'] = os.getcwd() + '/'
    default_dict['vor_target_snr'] = 20.0
    default_dict['exp_stamp'] = None
    return default_dict

def inspect_params(params):
    '''
    This function inspects the parameter dictionary
    :param params: parameter dictionary for inspection
    :return: a new directory with any missing keywords replaced to default values
    '''
    # a list of optional keyword arguments
    all_optional_kwargs = ['sigma','boxcar_smooth_size','forced_segvalues_galfitted','forced_segvalues_not_galfitted','inner_mask_semi_maj','inner_mask_semi_min','sep_sigma', 'sep_min_area','sep_dblend_nthresh',
                           'sep_dblend_cont','sep_filter_kwarg','sep_plot_destin','run_MC','plot_destin',
                           'SB_info_path','fits_save_loc', 'generate_voronoi_maps', 'vor_plot_save_loc', 'vor_fits_save_loc','vor_data_save_loc'
                           , 'vor_target_snr','exp_stamp','outer_phy_mask_size']
    # get the default parameter dictionary
    default_params = default_param_dict()

    # there are a bare minimum of 2 key words necessary. The galfit output file and the redshift.
    assert 'gfit_outfile' in params, "The line that mentions galfit output file is missing"
    assert 'redshift' in params, "The redshift key word is not found in the parameter file, consider adding it"

    for each_arg in all_optional_kwargs:# for each in optional keywords
        if params[each_arg] in ['none', 'None', 'NONE', ' ']:# if it is blank or none
            params[each_arg] = default_params[each_arg]# replace that keyword's value with the default value
        else:#otherwise pass..
            pass

    return params#return the updated parameter file.


def make_missing_dirs(list_of_dirs):
    '''
    In case the destination directories do not exist, this function helps you
    to make them.
    :param list_of_dirs: list of directories that need to be made.
    :return: nothing
    '''
    for each in list_of_dirs:# for each directory entry..
        if not os.path.exists(each): # if that path doesn't exist
            os.mkdir(each) # make it.
    return

def get_median_exp_time(exposure_stamp,feature):
    '''
    given an exposure map, this function will return a median
    exposure time value
    :param exposure_stamp: exposure map
    :param feature: (ignore, not used)
    :return: median exposure time
    '''
    exp_data = fits.getdata(exposure_stamp) # get the exposure time data
    # feature_exp_data = exp_data * feature
    median_exp_time = np.nanmedian(exp_data) #compute the median, while ignoring the nans
    return median_exp_time # return the median exposure time of the map.

def get_original_image_header(galfit_file_name):
    '''
    This function gets the header of the original image in a GALFIT cube
    :param galfit_file_name: galfit cube file name along with full path
    :return: header information
    '''
    image_cube = fits.open(galfit_file_name) #open the image cube
    original_header = image_cube[1].header #query the header
    image_cube.close() # close the image cube to prevent further i/o operations
    return original_header #return header

def get_model_header(gfit_name):
    '''
    Similar to the above function, but gets the header of the best-fit model image.
    This information has best-fit parameters for all the objects fit during GALFIT.
    :param gfit_name: galfit output file along with the full name
    :return: the model header.
    '''
    image_cube = fits.open(gfit_name) #load the image cube
    model_header = image_cube[2].header #grab the header
    image_cube.close() # close the cube to prevent further i/o operations
    return model_header # return model header.


'''Everything below this line is under construction. Future versions will include a Voronoi Tesselation of
the extracted features.
'''

def generate_data_for_voronoi(image_name,noise):
    loaded_image = fits.getdata(image_name)
    information_array = [['x','y','s','n']]
    y_shape, x_shape = loaded_image.shape
    for i in range(x_shape):
        for j in range(y_shape):
            if not np.isnan(loaded_image[j][i]):
                information_array.append([i,j,loaded_image[j][i],noise])
    return information_array


def make_vnr_image(bins_vr, vr_data,im_shape):
    vor_image = np.zeros(shape=im_shape)
    for i in range(len(bins_vr)):
        x = vr_data['x'][i]
        y = vr_data['y'][i]
        label = bins_vr[i]
        vor_image[y][x] = label
    return vor_image

def snr_map(bins_vr, vr_data, bin_snr,im_shape):
    snr_img = np.zeros(shape=im_shape)
    for i in range(len(bins_vr)):
        x = vr_data['x'][i]
        y = vr_data['y'][i]
        label = bins_vr[i]
        snr = bin_snr[label]
        snr_img[y][x] = snr
    return snr_img

def generate_voronoi_binning(feature_map,galfit_cube,vor_data,vor_tar_snr,plot_save_path,fits_save_path):
    import voronoi as vri
    image_name = feature_map
    original_image = fits.open(galfit_cube)[1].data

    csv_name = vor_data
    load_vri_data = np.genfromtxt(csv_name, dtype=None, names=True, delimiter=',')

    image_data = fits.getdata(image_name)
    xp = load_vri_data['x']
    yp = load_vri_data['y']
    sig = load_vri_data['s']
    noise = load_vri_data['n']

    tar_snr = vor_tar_snr

    pix_bin, bin_x, bin_y, bin_sn, bin_npix, scale = \
        vri.bin2d(xp.flatten(), yp.flatten(), sig, noise, tar_snr,
                  cvt=True, wvt=False, graphs=False, quiet=False)

    vri_image = make_vnr_image(pix_bin, load_vri_data, image_data.shape)
    snr_image = snr_map(pix_bin, load_vri_data, bin_sn, image_data.shape)

    snr_image[snr_image == 0] = np.nan
    norm = ImageNormalize(original_image, interval=PercentileInterval(99.),
                          stretch=AsinhStretch())
    norm2 = ImageNormalize(image_data, interval=PercentileInterval(99.),
                           stretch=AsinhStretch())
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[20, 8])
    ax1, ax2, ax3 = axs
    ax1.imshow(original_image, cmap='gray', origin='lower', norm=norm)
    ax2.imshow(original_image, cmap='gray', origin='lower', norm=norm)
    ax2.imshow(image_data, cmap='gist_heat', origin='lower', alpha=0.7, norm=norm2)
    # ax2.imshow(vri_image,cmap='nipy_spectral',origin='lower')
    ax3.imshow(original_image, cmap='gray', origin='lower', norm=norm)
    ax3.imshow(snr_image, cmap='nipy_spectral', origin='lower')
    make_axis_labels_off(axs)
    fig.tight_layout(h_pad=0.01, w_pad=0.01)
    plt.savefig(plot_save_path+'/%s.png' % os.path.basename(image_name).strip('.fits'), bbox_inches='tight')

    fits.writeto(fits_save_path+'/%s_vorBINS.fits' % os.path.basename(
        image_name).strip('.fits'), vri_image,overwrite=True)
    fits.writeto(fits_save_path+'/%s_vorSNR.fits' % os.path.basename(
        image_name).strip('.fits'), snr_image,overwrite=True)

    return
