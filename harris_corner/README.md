# HW1. Harris Corner Detection 

The function "harris_corner()" contain that gaussian_smooth(), sobel_x(), sobel_y, structure_tensor(), nms().
The sobel edge detection can use "sobel_edge_detection()".

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages

```bash

pip install os
pip install numpy
pip install scipy
pip install scikit-image
pip install matplotlib

```

## Usage

```bash

python3 harris_corner.py

```
## Main code

```python

# If you want to show each picture of processing. Otherwise, let it be 'None' or 'No' 
savefig = 'yes' 

#######################
## Sobel edge detect ##
#######################
direction_gradient, magnitude_gradient = sobel_edge_detection(img, Gaussian_size)

##########################
## Harris Corner detect ##
##########################
rr, cc = harris_corner(img_og, window=window, thresh=thresh, save=savefig)

# ============================================
# plot the corners ---------------------------
# ============================================
plt.imshow(img_og, cmap='gray')
plt.plot(cc, rr, 'r+', markersize='3')
plt.title('#Corners: {}'.format(rr.shape[0]))
plt.show()

# ============================================
# plot the sobel_edge ------------------------
# ============================================
fig2, ax2 = plt.subplots(1, 2, dpi=100)
ax2[0].imshow(magnitude_gradient)
ax2[1].imshow(direction_gradient)
plt.show()

```

## Reference

NTHU - 10810CS655000 - Computer Vision
