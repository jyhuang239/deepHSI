# -*- coding: utf-8 -*-
"""Training autoencoder-decoder model for HSI data cubes.
   https://github.com/jyhuang239/deepHSI.git
"""
#Importing Libraries 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
from auto_encoderdecoder import Autoencoder

torch.cuda.empty_cache()

# Define a combined loss function that includes MSE and gradient loss
def combined_loss(inpt_spectra, oupt_spectra, lambda_grad=0.5):
    # MSE loss for overall similarity
    mse_loss = nn.MSELoss()(oupt_spectra, inpt_spectra)
    
    # Gradient loss for sharp spectral features
    inpt_grad = torch.abs(inpt_spectra[:, :, 1:] - inpt_spectra[:, :, :-1])
    oupt_grad = torch.abs(oupt_spectra[:, :, 1:] - oupt_spectra[:, :, :-1])
    grad_loss = nn.L1Loss()(inpt_grad, oupt_grad)
    
    return mse_loss + lambda_grad * grad_loss

# Resample spectra with a given target_len
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

"""Read_tiff is a function that imports the hyperspectral SRS images then converts them to a numpy array."""
def read_tiff(path) :
    """
    path - Path of the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range (img.n_frames) : 
        img.seek(i)
        images.append(np.array(img))        
    return np.array(images)

# Importing the data, store it in to a numpy array
# This HSI cube contains six crop species: corn, cotton, sesame, broad-leaf soybean, narrow-leaf soybean, and rice. 
# The size of the imagery is 550 × 400 pixels, covering 270 spectral bands from 400 to 1000 nm.
HSIcube = "WHU-Hi-LongKou"
tif_path = f"./HSI_400-1000nm/{HSIcube}.tif"
print(f"Reading HSI file: {tif_path}")
inpt = read_tiff(tif_path) 
print(f"Shape of input data: {inpt.shape}")

#Reading min, max and mean value of the imported data
mean_input = np.mean(inpt)
max_input = np.max(inpt)
min_input = np.min(inpt)
differenc = max_input-min_input
print ('Mean value is = %f \n Max value is = %f \n Min value is = %f \n difference of max and min value is = %f ' % (mean_input,max_input,min_input, differenc))

inpt = (inpt - mean_input) / differenc

"""Reshaping input data, and then resample it to a new length"""
# Reshaping input data.
inpt = np.transpose(inpt, (1, 2, 0))

original_start = 400.0 
original_end = 1000.0 
n_frames = inpt.shape[2]  # original spectral length
target_len = 500 # new spectral length

# Define the old and new spectral axes
sp_old = np.linspace(original_start, original_end, n_frames)
sp_new = np.linspace(original_start, original_end, target_len)

resampled_data = vectorized_resample_spectra(inpt, sp_old, sp_new)
resampled_data = resampled_data.astype(np.float32)  # Ensure the resampled data is in float32 format
print(f"Shape of Resampled data: {resampled_data.shape} (dim_x, dim_y, target_len)")

data_N = resampled_data
del inpt, resampled_data

# Reshape it to img1d = dim_x*dim_y and then covert it to torch tensor. 
dim_x, dim_y, dim_z = data_N.shape

# Normalizing input data (mean value of 0, standard deviation of 1)
data_N = (data_N-np.mean(data_N))/(np.max(data_N)-np.min(data_N))

#Print min, max and mean value of the normalized input data
mean_input_n = np.mean(data_N)
max_input_n = np.amax(data_N)
min_input_n = np.amin(data_N)
differenc = max_input_n-min_input_n
print('Mean value is = %f \n Max value is = %f \n Min value is = %f \n difference of max and min value is = %f ' % (mean_input_n,max_input_n,min_input_n, differenc))

# Find the indices of the max voxel [xp_index, yp_index, zp_index]
# 1. Find the slice index of the max spectral peak 
slice_sums = np.sum(data_N, axis=(0, 1))
# Find the index of the maximum sum
posz = np.argmax(slice_sums)
# Get the actual maximum sum value
zp = slice_sums[posz]
print(f"The max slice sum is {zp}, which occurs at layer {posz}")

# 2. Find the y-pixel index at the max x-line sum
slice = data_N[:, :, posz]
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

# Ploting two slices of the HSI image.
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 10))

im1 = ax1.imshow(data_N[:,:,1],cmap='RdBu')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax2.imshow(data_N[:,:,posz],cmap='RdBu')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax1.set_title('Bkg: data at spec-channel 1', fontsize = 16)
ax2.set_title(f'Input data at spec-channel {posz}', fontsize = 16)
ax2.set_axis_off()
ax1.set_axis_off()

plt.show()

"""Reshape the numpy array, coverting it to pytorch tensor. Copy the tensor to GPU."""
#Converting numpy array to pyorch tensor and copy it on a GPU (CUDA).
input_t = data_N.reshape((dim_x*dim_y,-1))
input_t = torch.from_numpy(input_t).float()
input_t = input_t.view(-1,1,dim_z)
input_t = input_t.cuda()

#printing shape on input Torch tensor
print(f"Shape of input_t tensor: {input_t.shape}")

"""Using pytorch dataloader to make minibatch dataset for training.
70% of data used for training and 30% for validation.
"""
#Loading input_t data on the dataloader module.
n_train = int(len(input_t)*0.70)
n_test = int(len(input_t)) - n_train
train_data_set, test_data_set = torch.utils.data.random_split(input_t, [n_train, n_test])
train_loader  = torch.utils.data.DataLoader(train_data_set, batch_size = 256, shuffle=True, drop_last=True)
test_loader  = torch.utils.data.DataLoader(test_data_set, batch_size = 256, shuffle=True, drop_last=True)

#Now see how test/train loader works
print ("/////////////////////// Train dataset /////////////////////////")
for batch_idx, sample in enumerate(train_loader):
  inpt = sample
  print(f"Shape of inpt from train_loader: {inpt.shape}")
  
print ("/////////////////////// Test dataset /////////////////////////")
for batch_idx, sample in enumerate(test_loader):
  inpt = sample
  print(f"Shape of inpt from test_loader: {inpt.shape}")

#Printing the model architecture: dim_z / (2*2*2*2) = 32
print(Autoencoder(zdims=32, n_frames=dim_z))

# Defining hyper parameters of the model: number of epochs and learning rate
"""Defining model parameters and optimization method.
Learning rate = 0.001, scheduler "StepLR" used. 
Optimizer = Adam
Criterion = MSE
"""
learning_rate = 1e-3
num_epochs = 200
model = Autoencoder(zdims=32, n_frames=dim_z).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# training loop, scheduler learning was used to find the best learning rate. in evert 250 epoch learning rate changes by the factor of 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=390, gamma=0.7, last_epoch=-1)
model_save_name = f'HSI_{HSIcube}.pt'
model_path = f"./model/{model_save_name}" 

epoch_num = 0
train_error = []
test_error = []
best_error = 100
for epoch in range(num_epochs):
  print(scheduler.get_last_lr())
  loss_total = 0
  test_loss_total = 0
  model.train()
  for batch_idx, sample in enumerate(train_loader):
    inp = sample
    inp = inp.cuda()
    output = model(inp)
    loss = combined_loss(inp, output, lambda_grad=0.5)
    loss_total += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  loss_total = loss_total / (batch_idx+1)
  train_error.append(loss_total.item())

  model.eval()
  with torch.no_grad():
    for batch_idx, sample in enumerate(test_loader):
      inpt = sample
      inpt = inp.cuda()
      oupt = model(inpt)
      loss = combined_loss(inpt, oupt, lambda_grad=0.5) # criterion(oupt, inp)
      test_loss_total += loss
    test_loss_total = test_loss_total / (batch_idx+1)
    if loss < best_error:
      best_error = loss
      best_epoch = epoch
      print('Best loss at epoch', best_epoch)
      torch.save(model.state_dict(), model_path)
    test_error.append(test_loss_total.item())
    if epoch%10 == 9:
      epoch_num+=epoch_num
      print (('\r Train Epoch : {}/{} \tLoss : {:.4f}'.format (epoch+1,num_epochs,loss_total))) 
      print (('\r Test Epoch : {}/{} \tLoss : {:.4f}'.format (epoch+1,num_epochs,test_loss_total))) 

print('best loss', best_error, ' at epoch ', best_epoch)
plt.plot(train_error)
plt.plot(test_error)
plt.xlabel('Number of iteration')
plt.ylabel('Combined_Loss')
plt.title('Combined_Loss vs # of iterations')

#Ploting loss in log scale
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(train_error, label = "Error of training set")
ax.plot(test_error, label = "Error of test set")
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Combined_Loss (log Scale)")
ax.legend()
plt.show()