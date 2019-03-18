'''
This is the frontend to the Residual Feature Extraction pipeline.

A brief functioning of the pipeline:
This pipeline works on model-subtracted residual images by popular structural fitting routines (e.g., GALFIT, GIM2D, etc) to extract
faint low surface brightness features by isolating flux-wise and area-wise significant contiguous pixels regions by rigourous masking
routine. This routine accepts the image cubes (original image, model image, residual image) and generates several data products:

1. An Image with Extracted features.
2. Source extraction based segmentation map.
3. The background sky mask and the residual extraction mask.
4. A montecarlo approach based area threshold above which the extracted features are identified.
5. A catalog entry indicating the surface brightness and its error.

Author: Kameswara Bharadwaj Mantha
email: km4n6@mail.umkc.edu

Publication: Studying the Physical Properties of Tidal Features I. Extracting Morphological Substructure in CANDELS Observations and VELA Simulations.

Corresponding Author: Kameswara Bharadwaj Mantha
Co-authors: Daniel H. McIntosh, Cody P. Ciaschi, Rubyet Evan, Henry C. Ferguson, Logan B. Fries, Yicheng Guo, Luther D. Landry, Elizabeth J. McGrath,
Raymond C. Simons, Gregory F. Snyder, Scott E. Thompson, Eric F. Bell, Daniel Ceverino, Nimish P. Hathi, Anton M. Koekemoer,
Camilla Pacifici, Joel R. Primack, Marc Rafelski, Vicente Rodriguez-Gomez.
'''


################# Loading modules ##########

# regular modules to parse information
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.io import fits

# These two are the necessary modules for the pipeline to work.
from TF_main_functions import *
import MC_Athresh_finding as MC
###########################################

''' The parameter file is read in here and necessary values are extracted'''
parser = OptionParser()
parser.add_option("-p", "--param",action="store", type=None, dest="param_file",
                  help="Provide the parameter file with user-provided feature extraction options")
options, args = parser.parse_args()
parameters = parse_param_file(options.param_file)
''' The paramter file retrieval completed'''

'''
Inspecting the parameters to ensure all the necessary keywords are filled out.
In case, any values are left blank or none, then it will assume the default values
provided in the README file.
'''
inspected_params = inspect_params(parameters)

''' Here are list of necessary directories that the user provides for various information storage'''
list_of_dirs = [inspected_params['MC_path'],inspected_params['plot_destin'],inspected_params['SB_info_path'],
                inspected_params['sep_plot_destin'],inspected_params['fits_save_loc']]

make_missing_dirs(list_of_dirs)
''' 
Make the directories, if they don't exist. I recommend the users to create a specific organized data structure
and point them in the parameter file.
'''


'''Feature extraction starts here'''

surf_brightness_info=[] # initiating an empty sequence to store the surface brightness of the extracted features.

# Initiating the figure objects, to save each figure as a separate png file. Each for original, model, and the residuals.
fig = plt.figure(figsize=[10, 8])
ax = fig.gca()

fig2 = plt.figure(figsize=[10, 8])
ax2 = fig2.gca()

fig3 = plt.figure(figsize=[10, 8])
ax3 = fig3.gca()

# The galfit cube file provided in the parameter file.
galfit_file = inspected_params['gfit_outfile']
redshift = float(inspected_params['redshift']) # the redshift of the galaxy.
nyu_id = os.path.basename(galfit_file).strip('.fits') #stripping the extension to use the rest as an file name indicator.


''' Here is how the core functionality progresses:
1. Takes in the galfit file on which the feature extraction should take place.
2. The list of axes onto which the internal functions will do the plotting stuff.
3. redshift of the galaxy.
4. What is the flux-wise significance above the bkg sky should the sky pixels satisfy to be picked (as provided in the param file)
5. the annulus and masking mode is set to "intel", which stands for intelligent masking. There are some simplistic ones line circular, elliptical, which don't work that well.
Don't change it, unless you know what you are doing.
6. What is the 2D BoxCar kernel size to smooth the residual image with to boost the detection of faint features.
7. The dictionary that contains the inspected parameter file keywords and their respective values.
'''
original_data_can, tidal_feature_can, MAG_can, zp, PS, (feat_cen_x,feat_cen_y), what_ann_mode, local_bkg_mean, local_bkg_std, what_sigma, what_kernel, Jy_sec_per_electrons,rad_5kpc_line \
    = get_orig_and_feature(galfit_file,(ax,ax2,ax3,ax),redshift,float(inspected_params['sigma']),'intel',int(inspected_params['boxcar_smooth_size']),inspected_params)

# basically computes a binary mask of the extracted features, where pixels corresponding to the features are labelled as 1, and rest as 0.
spiky_feature_mask = get_feature_mask(tidal_feature_can)

'''
As described in the Mantha et al., 2019 manuscript, the feature extraction pipeline implements a MonteCarlo approach to
figuring out the optimum area threshold above which any detected features have less chance to be caused by noise fluctuations 
in the nearby spatial vicinity. This is an optional choice for the user. If set to True, then the pipeline will perform a MC analysis
and figure out the area threshold (Athresh). 

If the user has already performed a MC run previously and are changing other parameters to how the feature extraction result varies.
In such cases, setting it to false will make the code search for existing MC results from prior runs and will be automatically loaded. 

If any of existing results are not found, then based on the provided sigma (for now 1, 2 only supported) it will assume a default Athresh. 
Future updates will try to make this as a standalone parameter in the parameter file.
'''
if inspected_params['run_MC'].lower() == 'true':#checking if the MC run is set to true.
    print 'run_MC is set to True, running MC area distributions... this may take some time...'
    '''Takes in the parameter dictionary, local sky background value, sigma threshold of feature extraction, 
    how many MC iterations, smoothing box car kernel size, and 0.9999 indicating a 5sigma area significance.'''
    MC.mc_rand_area_cont(inspected_params,local_bkg_std,float(inspected_params['sigma']),200,float(inspected_params['boxcar_smooth_size']),0.9999) #generates a text file in the directory.
    a_threshold = np.loadtxt(inspected_params['MC_path'] + '/%s_Athresh_%ssig_%sker.txt'
                             % (nyu_id, float(inspected_params['sigma']), inspected_params['boxcar_smooth_size'])) # loads that file that it just created.
else:
    try:
        a_threshold = np.loadtxt(inspected_params['MC_path']+'/%s_Athresh_%ssig_%sker.txt'
                                 %(nyu_id,float(inspected_params['sigma']),inspected_params['boxcar_smooth_size'])) # try to see if there exists a text file from prior runs

    except IOError:#expect and accept an Ioerror, plausibly indicating that the file doesn't exist, then load default values.
        print 'The area threshold was not found... using Area Defaults'
        if what_sigma == 1.0: # if sigma is 1, then assume Athresh=60...
            a_threshold = 60
        elif what_sigma == 2.0: # if sigma is 2, then assume Athresh=20.
            a_threshold = 20

''' 
Providing the feature mask to localize the area-wise significant regions based on an area threshold and mapping these pixels to the
actual residual image.
'''
spiky_localized_features, spiky_feature_coords, spiky_feature_areas, spiky_feature_labels = area_based_spatial_localization(
            spiky_feature_mask, tidal_feature_can, a_threshold)

## making a plotting friendly array, where zeros are replaced with nans in the feature map, such that only the feature shows up.
spiky_feature_plt = make_TF_plotting_array(spiky_feature_coords, tidal_feature_can)

#Plot the feature, in heat map color over the residual image.
ax.imshow(spiky_feature_plt, origin='lower', alpha=0.7, cmap='gist_heat',
                  norm=SymLogNorm(0.0001, linscale=0.0001, vmin=0.0001, vmax=max(tidal_feature_can.flat) * 2))

# The user gets to provide an exposure stamp corresponding to the image, which will be used during the surface brightness
# calculation. If the user doesn't provide the data, an exposure time of 5000 seconds is assumed based on CANDELS observations
# as a default value.
if inspected_params['exp_stamp']!=None:
    median_exp_time = get_median_exp_time(inspected_params['exp_stamp'],spiky_feature_mask)
    feature_sb, d_feat_sb, phot_information_array = calculate_sb(spiky_localized_features, spiky_feature_areas, PS,
                                                                         local_bkg_mean, local_bkg_std,
                                                                         Jy_sec_per_electrons,median_exp_time)
elif inspected_params['exp_stamp']==None:
    feature_sb, d_feat_sb, phot_information_array = calculate_sb(spiky_localized_features, spiky_feature_areas, PS,
                                                                 local_bkg_mean, local_bkg_std,
                                                                 Jy_sec_per_electrons, 5000.0)
# for aesthetic purposes, the extracted feature is drawn as a contour, with solid linestyles and thick line widths.
#which can be changes as the user wishes in the following command.
draw_feat_contours(spiky_feature_coords, spiky_feature_mask.shape, ax,('-','-',4,3))
ax3.text(0.05, 0.9, '%s' % (nyu_id), transform=ax3.transAxes, color='white', fontsize=30) #print the file name for representation purposes.
ax.text(0.05, 0.9,
        '%s$\pm$%s mag arcsec$^{-2}$' % (round(feature_sb, 1), round(d_feat_sb, 1)),
        transform=ax.transAxes, color='white',
        fontsize=30) #print the surface brightness measured.

make_axis_labels_off(np.array([ax,ax2,ax3])) #make the axes labels off for aesthetic purposes.

zoom_in_image(ax,original_data_can.shape,(feat_cen_x,feat_cen_y),0.75) # zoom in on the image to 75% of its original size. Put it to 1, if you don't want to zoom.
zoom_in_image(ax2, original_data_can.shape,(feat_cen_x,feat_cen_y), 0.75)
zoom_in_image(ax3, original_data_can.shape,(feat_cen_x,feat_cen_y), 0.75)
xpos_line, y_pos_line = draw_kpc_scale(rad_5kpc_line, ax3, 0.98) # draw a 5 kpc line on the residual image.
ax3.text(xpos_line, y_pos_line * 0.96, '5 kpc', fontsize=30, color='white', ha='center', va='center') # say that it is 5 kpc line.


# save the figures in the plotting destination folder, res stands for residual image with feature overlaid on top.
# mod is the best-fit sersic model image provided in the cube. # org is the original image of the galaxy in the cube.
fig.savefig(inspected_params['plot_destin']+'/%s_%ssig_%sker_res.png'%(nyu_id,what_sigma,what_kernel),format='png',bbox_inches='tight')
fig2.savefig(inspected_params['plot_destin']+'/%s_%ssig_%sker_mod.png'%(nyu_id,what_sigma,what_kernel),format='png',bbox_inches='tight')
fig3.savefig(inspected_params['plot_destin']+'/%s_%ssig_%sker_org.png'%(nyu_id,what_sigma,what_kernel),format='png',bbox_inches='tight')

surf_brightness_info.append(['%s_%ssig_%sker'%(nyu_id,what_sigma,what_kernel),feature_sb,d_feat_sb]) # append the surface brightness information.
print '------\n %s finished'%(nyu_id)
# plt.show()

orig_img_header = get_original_image_header(galfit_file) # To preserve the wcs and header information, we query the header of the original image.
writeCsvFile(inspected_params['SB_info_path']+'/%s_%ssig_%sker_sbcat.csv'%(nyu_id,what_sigma,what_kernel),surf_brightness_info) # write a csv file containing the surface brightness information.

'''write the output of the feature extraction to a fits file, with the header information preserved from the original image.'''
fits.writeto(inspected_params['fits_save_loc']+'/%s_%ssig_%sker_feat_extract.fits'%(nyu_id,what_sigma,what_kernel),spiky_feature_plt,header=orig_img_header,overwrite=True)

''' This is an advanced feature. Make sure to have the VORNOI python pipeline installed before doing this. This is currently under implementation...
This will bin the extracted feature into equal bins of user-specified SNR using VORONOI tesselation. In principle, one can use the bin-by-bin information
to do semi-spatial analysis of these features.
'''
# for now, please set this to False, and in the future releases, this will be fully implemented.

if inspected_params['generate_voronoi_maps'].lower() == 'true': # if set to true, the vornoi pipeline will be run on the extracted features.
    voronoi_data = generate_data_for_voronoi(inspected_params['fits_save_loc']+'/%s_%ssig_%sker_feat_extract.fits'%(nyu_id,what_sigma,what_kernel),local_bkg_std) # generate data for voronoi tesselation map.
    writeCsvFile(inspected_params['vor_data_save_loc']+'/%s.csv'%(os.path.basename(galfit_file).strip('.fits')),voronoi_data) # write the output to a csv file.
    generate_voronoi_binning(inspected_params['fits_save_loc']+'/%s_%ssig_%sker_feat_extract.fits'%(nyu_id,what_sigma,what_kernel),
                             galfit_file,inspected_params['vor_data_save_loc']+'/%s.csv'%(os.path.basename(galfit_file).strip('.fits'))
                             ,float(inspected_params['vor_target_snr']),inspected_params['vor_plot_save_loc'],
                             inspected_params['vor_fits_save_loc']) # generate the voronoi maps and save the respective in the folders.