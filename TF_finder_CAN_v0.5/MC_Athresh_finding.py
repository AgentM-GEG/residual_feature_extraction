'''
This function carries out an area threshold calculation using a MonteCarlo approach. An area threhsold is defined as the area
above which the contiguous pixel regions have a minimal chance of being caused by a spatially-correlated noise spike.

This function plays an important role in the Tidal feature finder pipeline.

Author: Kameswara Bharadwaj Mantha
email: km4n6@mail.umkc.edu

'''

###Load the modules
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.convolution import Box2DKernel, convolve
from astropy.modeling import models, fitting
from skimage import measure
import os
####################

def smooth_tidal_feature(feat,box_size):
    '''
    Smoothing the input image using a Boxcar filter.
    :param feat: input image to be smoothed
    :param box_size: size of the boxcar filter
    :return: smoothed image
    '''
    from astropy.convolution import Box2DKernel, convolve #import modules
    kernel = Box2DKernel(box_size) # initiate the kernel
    convolved_feature = convolve(feat,kernel) #convolve the image with the kernel
    return convolved_feature # return the smoothed feature.


def sgm_mask(data,clip_lim):
    '''
    Creates an image with pixels satisfying a flux-wise contiguous regions.
    :param data: image data
    :param clip_lim: what is the limit above which pixels need to be chosen.
    :return: feature image
    '''
    copy_data = np.zeros(shape=data.shape)# create an image of zeroes
    y_shape,x_shape = data.shape #get the dimensions of the image data
    for x in range(0,x_shape,1):
        for y in range(0,y_shape,1):#basically iterating over each pixel
            if data[y][x] >=clip_lim:# if the pixel value is above the limit
                copy_data[y][x] =data[y][x] #then copy the value in the image data.
            else:
                pass
    return copy_data



def feat_mask(feat):
    '''
    Given a image, everything else above zero is made ONE.
    :param feat: input image
    :return: a binary mask
    '''
    null_array = np.zeros(shape=feat.shape) #create a new image array of zeroes
    x_shape, y_shape = feat.shape #get the shape of the image.
    for i in range(x_shape):
        for j in range(y_shape):#iterate over each pixel.
            if feat[i][j]!=0:
                null_array[i][j] =1 # make every pixel above zero to be 1.
    return null_array # return the binary mask.


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


def mc_rand_area_cont(params,local_bkg_value,nsigma, num_mc_iters,kernel_size,percentile):
    '''
    This function computes an area threshold above which the contiguous regions caused by
    random noise induced fluctuations is minimal.
    :param params: paramter file.
    :param local_bkg_value: the 1 sigma standard deviation of the best-fit gaussian to the background sky.
    :param nsigma: what is the significance above the background sky sigma should the pixels be chosen.
    :param num_mc_iters: number of montecarlo iterations
    :param kernel_size: what is the box car kernel size with which the image is smoothed.
    :param percentile: what percentile level. 99.99% means, there is 0.01 chance that a contiguous region of an Area A
    is occured due to random chance fluctutation.
    :return: Nothing.
    '''
    galfit_file = params['gfit_outfile'] #what is the name and path of the GALFIT output file
    gal_id = os.path.basename(galfit_file).strip('.fits') # id for file name

    cube = fits.open(galfit_file) #open the galfit cube
    data_org = cube[1].data# get the original data
    cube.close() # close the cube.

    all_areas = [] #initiate an empty area list in which the random contiguous pixel areas are populated.
    random_pick_iters = np.random.choice(np.arange(1,num_mc_iters,1),1) #randomly pick one iteration to save a figure.
    for iterations in np.arange(1,num_mc_iters,1):# for each MC itearation
        noise = np.random.normal(0.0, local_bkg_value, data_org.shape) # generate a gaussian noise image
        kernel = Box2DKernel(kernel_size) # initiate a boxcar kernel
        convovled_noise = convolve(noise, kernel) # smooth the noise image with the kernel

        # generate a histogram
        hist, bins = np.histogram(np.array([i for i in convovled_noise.flat if i != 0]), bins=100, normed=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        #fitting a gaussian
        g_init = models.Gaussian1D(amplitude=np.max(hist), mean=0, stddev=local_bkg_value)
        fit_g = fitting.LevMarLSQFitter()
        gfit = fit_g(g_init, bin_centers[:np.argmax(hist) + 10], hist[:np.argmax(hist) + 10])

        # now, isolate the pixels above the user specified significance
        spiked_pix = sgm_mask(convovled_noise, nsigma * abs(gfit.stddev.value))

        # make a binary image to find contiguous regions.
        for_label_detection = feat_mask(spiked_pix)

        # find the pixel-contiguity.
        labels = measure.label(for_label_detection, background=0)
        label_props = measure.regionprops(labels, intensity_image=data_org) # get the properties of the contiguous regions

        random_area_contribution = np.array([i.area for i in label_props]) # make an array of areas
        all_areas.append(random_area_contribution) # append that to the all areas list.
        if iterations in random_pick_iters: # for the iteration that was randomly chosen.
            # make the plot illustrating the whole process above
            iteration_destin = params['MC_path']
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[24, 12])
            [ex1, ex3] = axs
            ex1.imshow(convovled_noise, cmap='gray',origin='lower',interpolation='nearest')

            ex3.imshow(labels, cmap='nipy_spectral',origin='lower')

            ex1.text(0.05,0.03, 'Noise Image', transform = ex1.transAxes, fontsize=28,color='white')
            # ex2.text(0.05,0.03, 'Masked Noise Image', transform = ex2.transAxes, fontsize=28,color='white')
            ex3.text(0.05,0.03, 'Contiguous Pixel Regions', transform = ex3.transAxes, fontsize=28,color='white')

            make_axis_labels_off(axs)
            plt.subplots_adjust(wspace=0)
            plt.tight_layout()
            plt.savefig(iteration_destin + '/%s_iter%s.png' % (gal_id,iterations), bbox_inches='tight')
            plt.cla()
            plt.close(fig)

    # save an numpy array file with all the areas calculated in the MC iterations.
    np.save(params['MC_path']+'/%s_areas_%ssig_%sker.npy' % (gal_id,nsigma,int(kernel_size)) ,all_areas)

    # compute a cumulative histogram of the areas.
    hist_data = np.cumsum(np.histogram(np.concatenate(all_areas),np.arange(1,75,1),density=True)[0])
    index_array = np.arange(len(hist_data))
    a_thresh = min(index_array[hist_data>percentile]) # choose the area at which the cumulative histogram hits the percentile value.

    np.savetxt(params['MC_path']+'/%s_Athresh_%ssig_%sker.txt'%(gal_id,nsigma,int(kernel_size)),np.array([int(a_thresh)]))
    # save a text file with the area value computed in the above step.
    return
