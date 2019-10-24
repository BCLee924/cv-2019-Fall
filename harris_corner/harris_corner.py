#!/usr/bin/env python3
#
##--------- [hw1_1] Harris Corner Detection  -------------
# * Practice
# * Author: Colin, Lee - 108134506
# * Date: Oct 18th, 2019
##--------------------------------------------------------
#
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from scipy import ndimage as ndi

from skimage import io
from skimage import util
from skimage import color
from skimage import transform

#====================================================================
output_folder_save_path = os.getcwd()+'/results/'
#====================================================================
img = io.imread('original.jpg')
# img = io.imread('test.jpg')
img = util.img_as_float(img)
img_gray = color.rgb2gray(img)
#====================================================================

def gaussian_smooth(img, gsize, save=None):
    s = 5
    cx, cy = (0+gsize-1)/2, (0+gsize-1)/2
    xx, yy = np.meshgrid(np.arange(gsize) - cx, np.arange(gsize) -cy)
    #gaussian form
    g = np.exp(-(xx **2 + yy **2 ) / (2 * (s **2))) / (2 * np.pi * (s **2))
    g = g / g.sum()
    #convolution
    output = signal.convolve2d(img, g, mode='same')
    #save it or not
    if save == 'yes':
        plt.figure()
        plt.imshow(output, cmap='gray')
        plt.title('Gassian size: {}'.format(gsize))
        plt.savefig(output_folder_save_path+'GaussianBlur_size_{}.png'.format(gsize))
    return output

def sobel_x(img):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

    kernel = kernel #/ 8
    dx = signal.convolve2d(img, kernel, mode='same')
    return dx

def sobel_y(img):
    kernel = np.array([[1,   2,  1],
                       [0,   0,  0],
                       [-1, -2, -1]])

    kernel = kernel #/ 8
    dy = signal.convolve2d(img, kernel, mode='same')
    return dy

def sobel_edge_detection(gray, Gsize,save=None):
    ## Find the dx, dy first!!##
    dx = sobel_x(gray)
    dy = sobel_y(gray)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    h, w = gray.shape

    hsv = np.zeros((h, w, 3))
    hsv[...,0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    hsv[...,1] = 0.
    hsv[...,2] = (mag - mag.min()) / (mag.max() - mag.min())

    # hsv[...,0] = np.pi
    G_mag = color.hsv2rgb(hsv)

    # hsv[...,0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    # hsv[...,2] = 0.8
    # G_dir = color.hsv2rgb(hsv)

    hsv[...,0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    hsv[...,1] = 1.
    hsv[...,2] = (mag - mag.min()) / (mag.max() - mag.min())
    rgb = color.hsv2rgb(hsv)
    #save it or not
    if save == 'yes':
        plt.figure()
        plt.imshow(G_mag);plt.title('Magnitude gradient')
        plt.savefig(output_folder_save_path+'MagnitudeGradient_Gsize_{}.png'.format(Gsize))
        plt.figure()
        plt.imshow(rgb);plt.title('Direction Gradient')
        plt.savefig(output_folder_save_path+'DirectionGradient_Gsize_{}.png'.format(Gsize))

    return rgb, G_mag#, G_dir

def structure_matrix(dx, dy, window=30):
    Axx = gaussian_smooth(dx * dx, window)
    # Axx = signal.convolve2d(dx * dx, np.ones((window, window)),mode='same')
    Axy = gaussian_smooth(dx * dy, window)
    # Axy = signal.convolve2d(dx * dy, np.ones((window, window)),mode='same')
    Ayy = gaussian_smooth(dy * dy, window)
    # Ayy = signal.convolve2d(dy * dy, np.ones((window, window)),mode='same')

    return Axx, Axy, Ayy

def harris_response(Axx, Axy, Ayy, k=0.05):
    det = Axx * Ayy - Axy * Axy
    tr  = Axx + Axy
    R   = det - k * tr * tr
    return R

def nms(R, window=10, thresh=3e-7):
    mask1 = (R > thresh)
    mask2 = (np.abs(ndi.maximum_filter(R, size=window) - R) < 1e-6)
    mask = (mask1 & mask2)
    return np.nonzero(mask)

def local_max(R, window=10, thresh=1e-7):
    maxR = ndi.maximum_filter(R, size=window)
    mask1 = np.abs(maxR - R) < 1e-6
    mask2 = (R > thresh)
    return np.nonzero(mask1 & mask2)

## Harris Corner ##
def harris_corner(img, gsize=10, k=0.05, window=30, thresh=3e-7, save=None):
    img = gaussian_smooth(img, gsize)
    dx = sobel_x(img)
    dy = sobel_y(img)
    Axx, Axy, Ayy = structure_matrix(dx, dy, window)
    R = harris_response(Axx, Axy, Ayy, k=k)
    # rr, cc = local_max(R, window=window, thresh=thresh)
    rr, cc = nms(R, window=window, thresh=thresh)

    if save == 'yes':
        rr1, cc1 = nms(R, window=3, thresh=thresh)
        rr2, cc2 = nms(R, window=30, thresh=thresh)
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(cc1, rr1, 'r+', markersize='3')
        plt.title('Window size: 3 , #Corners: {}'.format(rr1.shape[0]))
        plt.savefig(output_folder_save_path+'Result_size_3.png')
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(cc2, rr2, 'r+', markersize='3')
        plt.title('Window size: 30, #Corners: {}'.format(rr2.shape[0]))
        # plt.savefig(output_folder_save_path+'Result_size_30.png')

    return rr, cc

if __name__ == '__main__':
    img_og = img_gray
    img_ro = transform.rotate(img_og, 30.0)
    img_sc = transform.rescale(img_og, 0.5, multichannel=False)

    gsize  = [5, 10] #5
    window = 30
    thresh = 1e-4#8e-5#5e-5

    savefig = 'yes'

    #plot to compare different size of the gaussian kernel-------
    fig, ax = plt.subplots(1, 2, dpi=200)
    fig1, ax1 = plt.subplots(1, 2, dpi=200)
    fig2, ax2 = plt.subplots(1, 2, dpi=200)
    counter = 0
    for i in range(2):
        k = gsize[i]
        img_afterGaussian = gaussian_smooth(img_og, k)
        G_dir, G_mag= sobel_edge_detection(img_afterGaussian, k, savefig)
        ax[counter].imshow(img_afterGaussian, cmap='gray')
        ax[counter].set_title('Gaussian size: {}'.format(k))
        ax1[counter].imshow(G_mag);ax1[counter].set_title('Gaussian size: {}'.format(k))
        ax2[counter].imshow(G_dir);ax2[counter].set_title('Gaussian size: {}'.format(k))
        counter += 1
    fig.savefig(output_folder_save_path+'GaussianBlur.png')
    fig1.savefig(output_folder_save_path+'MagnitudeGradient.png')
    fig2.savefig(output_folder_save_path+'DirectionGradient.png')

    ############################### Harris Corner ####################################
    rr1, cc1 = harris_corner(img_og, window=window, thresh=thresh)
    rr2, cc2 = harris_corner(img_ro, window=window, thresh=thresh)
    rr3, cc3 = harris_corner(img_sc, window=window, thresh=thresh)
    #plot the corners ---------------------------
    corner_fig, corner_ax = plt.subplots(1, 3, dpi=100, figsize=(10,8))
    corner_ax[0].imshow(img_og, cmap='gray')
    corner_ax[1].imshow(img_ro, cmap='gray')
    corner_ax[2].imshow(img_sc, cmap='gray')
    corner_ax[0].plot(cc1, rr1, 'r+', markersize='3')
    corner_ax[1].plot(cc2, rr2, 'r+', markersize='3')
    corner_ax[2].plot(cc3, rr3, 'r+', markersize='3')
    corner_ax[0].set_title('#Corner: {}'.format(rr1.shape[0]))
    corner_ax[1].set_title('#Corner: {}'.format(rr2.shape[0]))
    corner_ax[2].set_title('#Corner: {}'.format(rr3.shape[0]))
    corner_fig.tight_layout()
    # plt.title('Original Rotation Scaled')
    corner_fig.savefig(output_folder_save_path+'Og_Ro_Sc_compare.png')


    ################## Different window size of structure_tensor ####################
    rr1, cc1 = harris_corner(img_og, window=3, thresh=thresh)
    rr2, cc2 = harris_corner(img_og, window=15, thresh=thresh)
    rr3, cc3 = harris_corner(img_og, window=30, thresh=thresh, save=savefig) #save the figures of 3, 30 wsize
    #plot the corners ---------------------------
    win_fig, win_ax = plt.subplots(1, 3, dpi=100, figsize=(10,8))
    win_ax[0].imshow(img_og, cmap='gray')
    win_ax[1].imshow(img_og, cmap='gray')
    win_ax[2].imshow(img_og, cmap='gray')
    win_ax[0].plot(cc1, rr1, 'r+', markersize='3')
    win_ax[1].plot(cc2, rr2, 'r+', markersize='3')
    win_ax[2].plot(cc3, rr3, 'r+', markersize='3')
    win_ax[0].set_title('#W3__Corner: {}'.format(rr1.shape[0]))
    win_ax[1].set_title('#W15_Corner: {}'.format(rr2.shape[0]))
    win_ax[2].set_title('#W30_Corner: {}'.format(rr3.shape[0]))
    # plt.title('window size')
    win_fig.tight_layout()
    win_fig.savefig(output_folder_save_path+'3_15_30_compare.png')

    ##show the images
    plt.show()
