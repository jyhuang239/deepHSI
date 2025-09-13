# -*- coding: utf-8 -*-
"""Running autoencoder-decoder (AED) model on HSI data cubes for image segmentation based on
spectral features.
    https://github.com/jyhuang239/deepHSI.git
"""

#Importing Libraries 
from os import path
import numpy as np
import pandas as pd
from PIL import Image

import torch
#from scipy import ndimage 
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from auto_encoderdecoder import Autoencoder

torch.cuda.empty_cache()

#Read_tiff(path) reads a tiff image stack and return it to a numpy array.
def read_tiff(path) :
    img = Image.open(path)
    images = []
    for i in range (img.n_frames) : 
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

# Resample spectra to an specified spectral length
def vectorized_resample_spectra(data, z_old, z_new):
    """
    Resamples a 3D data cube using a vectorized approach.

    This function is significantly more efficient than iterating through the array.
    The `interp1d` function can operate on multi-dimensional arrays if you specify
    the axis along which to interpolate.

    Args:
        data (np.ndarray): The input data with shape (dimx, dimy, spectral_points_old).
        z_old (np.ndarray): The original spectral axis coordinates.
        z_new (np.ndarray): The new spectral axis coordinates to interpolate onto.

    Returns:
        np.ndarray: The resampled data with shape (dimx, dimy, spectral_points_new).
    """
    # Create an interpolation function for the entire data cube at once.
    # The `axis=-1` argument tells interp1d to treat the last dimension of `data`
    # as the axis to interpolate along. This creates a set of interpolation
    # functions for each spectrum in a highly optimized way.
    interpolator = interp1d(z_old, data, 
                            kind='linear', bounds_error=False, 
                            fill_value='extrapolate', axis=-1)
    
    # Apply the interpolator to the new spectral axis.
    # This operation is broadcast across all the (i, j) spectra, performing
    # the interpolation for the entire cube in a single, efficient call.
    resampled_data = interpolator(z_new)
    
    return resampled_data

# Importing the data, store it in to a numpy array
# This HSI cube contains six crop species: corn, cotton, sesame, broad-leaf soybean, narrow-leaf soybean, and rice. 
# The size of the imagery is 550 × 400 pixels, covering 270 spectral bands from 400 to 1000 nm.
HSIcube = "WHU-Hi-LongKou"
tif_path = f"./HSI_400-1000nm/{HSIcube}.tif"
print(f"Reading HSI file: {tif_path}")
imgStk = read_tiff(tif_path) 
#printing shape of the imgData
nframes, dim_x, dim_y = imgStk.shape
print (f"Shape of imgStk: {nframes} x {dim_x} x {dim_y}")

#Finding min, max and mean value of the imported data
mean_input = np.mean(imgStk)
max_input = np.max(imgStk)
min_input = np.min(imgStk)
differenc = max_input - min_input
print("Input data statistical features:\n")
print("//////////////////////////////////////")
print('\tMean value is = %f \n \tMax value is = %f \n \t Min value is = %f \n \tDifference of max and min value is = %f ' % (mean_input,max_input,min_input, differenc))

#Normalizing input data
imgStk = (imgStk - mean_input) / (max_input - min_input)
#Check the data normalization
print(" ")
print("Check the data normalization result")
print('\tMean value is = %f \n \tMax value is = %f \n \t Min value is = %f \n \tDifference of max and min value is = %f ' 
      % (np.mean(imgStk),np.max(imgStk),np.min(imgStk), np.max(imgStk) - np.min(imgStk)))

# Reshaping input data.
inpt = np.transpose(imgStk, (1, 2, 0))

# Resample spectra to target_len
original_start = 400.0 
original_end = 1000.0 
n_frames = inpt.shape[2]  # original spectral length
target_len = 500 # new spectral length

# Define the old and new spectral axes
sp_old = np.linspace(original_start, original_end, n_frames)
sp_new = np.linspace(original_start, original_end, target_len)

input = vectorized_resample_spectra(inpt, sp_old, sp_new)
input = input.astype(np.float32)  # Ensure the resampled data is in float32 format
print(f"Shape of Resampled data: {input.shape} (dim_x, dim_y, target_len)")
del imgStk, inpt

# Find the max voxel index [xp_index, yp_index, zp_index]
# 1. Find the slice index at the max spectral peak 
slice_sums = np.sum(input, axis=(0, 1))
# Find the index of the maximum sum
posz = np.argmax(slice_sums)
# Get the actual maximum sum value
zp = slice_sums[posz]
print(f"The max slice sum is {zp}, which occurs at layer {posz}")

# 2. Find the y-pixel index at the max x-line sum
slice = input[:, :, posz]
x_sums = np.sum(slice, axis=0)
# Find the index of the maximum sum
posy = np.argmax(x_sums)
# Get the actual maximum sum value
yp = x_sums[posy]
print(f"The max x-line sum is {yp}, which occurs at y-pixel {posy}")

# 3. Find the x-pixel index at the max y-line sum
y_sums = np.sum(slice, axis=1)
# Find the index of the maximum sum
posx = np.argmax(y_sums)
# Get the actual maximum sum value
xp = x_sums[posx]
print(f"The max y-line sum is {xp}, which occurs at x-pixel {posx}")

# Ploting the slice images of the input data cube
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))

im1 = ax1.imshow(input[:,:,1],cmap='RdBu')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax2.imshow(input[:,:,posz],cmap='RdBu')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax1.set_title('Bkg: data at spec-channel 1', fontsize = 16)
ax2.set_title(f'Input data at spec-channel {posz}', fontsize = 16)
ax2.set_axis_off()
ax1.set_axis_off()

# Reshape it to img1d = dim_x*dim_y and then covert it to torch tensor. 
dim_x, dim_y, dim_z = input.shape

img1d = dim_x*dim_y
input_t = input.reshape((img1d,-1))

input_t = torch.from_numpy(input_t).float()
input_t = input_t.view(-1,1,dim_z)
print(f"Shape of input_t = {input_t.shape}")

#Output of the NN
outpt = torch.zeros([img1d,1,dim_z],dtype=torch.float)

#loading the model parameters
model = Autoencoder(zdims=32, n_frames=dim_z)
model_save_name = f'HSI_{HSIcube}.pt'
model_path = f"./model/{model_save_name}" 
model.load_state_dict(torch.load(model_path, map_location='cpu'))

# For classifiction, we are going to take advantage of Latent space. 
# This requires to feed in all input data to just Encoder layer and then applies one of 
# clustering method to label every single pixel.
# projecting input data to latent space for the purpose of classification. 
model.eval() # Set model to evaluation mode
with torch.no_grad():
    outpt = model(input_t)
    print(f"Shape of model output: {outpt.shape}")

# Reshape the outpt Torch tensor, and make it an image format.
outpt = outpt.view(img1d, dim_z)
outpt = outpt.detach().cpu().numpy()
outpt = outpt.reshape(dim_x,dim_y,-1)
print(f"Shape of outpt = {outpt.shape}")

# Display the statistics of outpt tensor 
max_outpt = np.max(outpt)
min_outpt = np.min(outpt)
mean_outpt = np.mean(outpt)

print("Output data statistical features:")
print("//////////////////////////////////////\n")
print ('\tmean = %f \n \tmax = %f \n \tmin = %f ' % (mean_outpt,max_outpt,min_outpt))
print('\tVariance of the Output data:',max_outpt-min_outpt)

"""Compare the AE model output with the input image data."""
# plot a spectral-peak (posz) frame of the input data cube
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8, 8))

im1 = ax1.imshow(input[:,:,posz],cmap='RdBu', vmin=-0.4, vmax=0.4)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.08)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax2.imshow(outpt[:,:,posz], cmap='RdBu', vmin=-0.3, vmax=0.3)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax1.set_title(f'Input @ sp={posz}', fontsize = 12)
ax2.set_title(f'AED Output @ sp={posz}', fontsize = 12)
ax2.set_axis_off()
ax1.set_axis_off()

plt.show()

"""Checking spectral reconstruction. Every single pixel contains full spectrum information. """
# x represents spectral axis.
x = np.linspace(original_start,original_end,dim_z)

# Ploting a single pixel spectrum of the reconstructed data.
plt.figure(figsize=(8, 6))
plt.plot(x, outpt[posx,posy,:],color = 'r',alpha=0.8,linewidth=2,label='AED-output')

# Ploting a single-pixel spectrum of the input data on top of each other
plt.plot(x, input[posx,posy,:],color = 'g',alpha=0.6,label='Input')
plt.tick_params(width = 2,length = 4,direction = "in")
plt.legend(prop={'size': 12},shadow=True, bbox_to_anchor=(0.5, 0.8))
plt.xticks(fontsize=14)
y_ticks = np.arange(-0.2, 1.1, 0.4)
plt.yticks(y_ticks,fontsize = 12)
plt.ylabel('Normalized Intensity',fontsize = 14)
plt.xlabel('Wavelength (nm)', fontsize = 14)
plt.title(f'Spectrum at Pixel ({posx},{posy})', fontsize = 14)
plt.show()

model.eval() # Set model to evaluation mode
with torch.no_grad():
    # We only need the latent vector 'z' for clustering
    latent_space, _ = model.encode(input_t) 
    latent_space = latent_space.cpu().numpy()
    # checking latent space shape.
    print(f"Shape of latent space = {latent_space.shape}")

# 1. Scale the Data
scaler = StandardScaler()
latent_space_scaled = scaler.fit_transform(latent_space)
del latent_space

# 2. Perform GMM Clustering
n_clusters = 9
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
clusters = gmm.fit_predict(latent_space_scaled)

# Get cluster counts:
cluster_counts = np.bincount(clusters)
print(f"Data points per cluster: {cluster_counts}")

# 3. Get Cluster Gravity Centers (Means)
# These are the "gravity centers" of the Gaussian components
cluster_centers_scaled = gmm.means_

# 4. Inverse-transform the centers back to the original latent space scale
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

# 5. Decode the Representative Spectra from the Centers
with torch.no_grad():
    # Convert centers to a tensor and move to the model's device
    centers_tensor = torch.from_numpy(cluster_centers_original).float()
    
    # Use the modified decode method (input_sizes is not needed)
    representative_spectra = model.decode(centers_tensor)
    representative_spectra = representative_spectra.cpu().numpy()

# 6. Plot the Representative Spectra
fig, axs = plt.subplots(n_clusters, 1, constrained_layout=True)

# Plot each graph, and manually set the y tick values
for i in range(n_clusters):
    axs[i].plot(x, representative_spectra[i].squeeze(), label=f'Cluster {i+1} Center')
    plt.legend(fontsize=10)
    axs[i].set_xlim(np.min(x), np.max(x))

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.show()

#Compare the segmented image with input image
#Pandas converted to Numpy array, every single pixel is color coded.
latent_space = pd.DataFrame(latent_space_scaled)
latent_space["Clusters"] = clusters
Latet_cluster_img = latent_space["Clusters"].to_numpy()
Latet_cluster_img = np.reshape(Latet_cluster_img,(dim_x,dim_y))
print("Shape of the color coded image is:", Latet_cluster_img.shape)

from matplotlib.colors import ListedColormap
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))
cmap = ListedColormap(["gray", "green","blue","orange","red", "yellow", "cyan", "magenta", "white"])
im1 = ax1.imshow(Latet_cluster_img, cmap = cmap)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax2.imshow(input[:,:,posz],cmap='RdBu', vmin=-0.4, vmax=0.4)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax1.set_title('Image with 5 GMM Clusters', fontsize = 16)
ax2.set_title(f'Model Output @ sp={posz}', fontsize = 16)
ax2.set_axis_off()
ax1.set_axis_off()

plt.show()
