import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
import os
##########################################################################
# Function definitions

def read_velocity_data(input_dir, filename, field_name, reader, file_ext, t_1, t_n):
# Read velocity data from file
# Inputs:
# input_dir - input directory location
# filename - velocity timeseries filename 
# field_name - velocity field name in the vtk files
# reader - vtk reader
# file_ext - velocity data file type - either vtk or vtu
# t_1 - first timestep to read
# t_n - last timestep to read
# Outputs:
# X - data matrix containing the velocity data
# mesh - mesh object containing the mesh

    print('Reading velocity data and mesh from:', input_dir + filename, flush = True)

    velocity_list = []
    for i in range(t_1,t_n,1):
        reader.SetFileName(input_dir+filename+str(i)+ file_ext)
        reader.Update()
        output = reader.GetOutput()
        velocity_dataset = output.GetPointData().GetArray(field_name)
        velocity = VN.vtk_to_numpy(velocity_dataset)
        velocity_vec = np.reshape(velocity,(-1,1))
        velocity_list.append(velocity_vec)

    # arrange the velocity data into a big data matrix
    X = np.asarray(velocity_list)
    X = X.flatten('F')

    X = np.reshape(X,(-1,t_n-t_1))
    # rows of X correspond to velocity components at spatial locations
    # columns of X correspond to timesteps
    #     t_1 t_2.  .  t_n
    # X = [u  u  .  .  .]  (x_1,y_1)
    #     [v  v  .  .  .]  (x_1,y_1)
    #     [w  w  .  .  .]  (x_1,y_1)
    #     [u  u  .  .  .]  (x_2,y_2)
    #     [v  v  .  .  .]  (x_2,y_2) 
    #     [w  w  .  .  .]  (x_2,y_2)
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .

    # read the mesh for later visualization and saving data
    mesh = reader.GetOutput()

    return X, mesh


def convertToMagnitude(X):
# Use velocity magnitude instead of the vector   
# Input:
# X - original data matrix with velocity vector
# Output:
# X_mag - velocity data matrix containing velocity magnitude 
#     t_1   t_2  .  .  t_n
# X_mag = [|u|  |u|  .  .  .]  (x_1,y_1)
#         [|u|  |u|  .  .  .]  (x_2,y_2)
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .

    n = X.shape[0]
    m = X.shape[1]
    X_mag = np.zeros((int(n/3),m))

    for i in range(0,m):
        Ui = X[:,i]
        Ui = np.reshape(Ui,(-1,3))
        Ui_mag = np.sqrt(np.sum(np.square(Ui),1))
        X_mag[:,i] = Ui_mag

    return X_mag



def subtract_mean(X):
# subtract the temporal mean of the data set
# Input:
# X - original data matrix
# Output:
# X - data matrix with temporal mean subtracted
# X_mean - temporal mean of the data
    n = X.shape[0]
    m = X.shape[1]  
    X_mean = np.mean(X,1)
    for i in range(0,n):
        X[i,:] = X[i,:]-X_mean[i]

    X = (1/np.sqrt(m)* X)
    return X, X_mean



def save_modes(u, mesh, output_file, modesToSave, field_name):
# Save the spatial modes to a VTK file for visualization
# Input:
# u - matrix containing the eigenvectors
# mesh - mesh object containing the location of the mesh points
# output_file - name and location of the output vtk file for the modes
# modesToSave - number of modes to save to the VTK file
# field_name - velocity field name in the original vtk files (just for removing unnecessary data)


    mesh = dsa.WrapDataObject(mesh)

    if modesToSave > u.shape[1]:
        modesToSave = u.shape[1]
        print('Max number of modes is', u.shape[1])
    
    print('Saving the first',modesToSave, 'PCA modes to ',out_filename)

    for i in range(0,modesToSave,1):
        U_i = u[:,i]
        mesh.PointData.append(U_i, 'mode_'+str(i))
        mesh.PointData.RemoveArray(field_name) 
        if i == 0 and subtract_mean_flag:
            mesh.PointData.append(X_mean, 'U_mean')
        if case == 'turbulent_channel/':
            writer = vtk.vtkRectilinearGridWriter()
        else:
            writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(mesh.VTKObject)
        writer.Write()


def plotSpectrum(s):
# Plot singular value spectrum
# Input:
# s - vector containing the singular values
    sigma = s
    sigma_energy = np.cumsum(sigma)
    # plot singular values and the cumulative energy
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10, 6), dpi=80)
    #plot only up to the first 100 singular values
    ax1.plot(sigma[0:99],'ko',markersize=3)
    ax1.set_xlabel('Modes')
    ax1.set_ylabel('Singular values')
    ax1.set_title('PCA')

    ax2.plot(sigma[0:99],'ko',markersize=3)
    ax2.semilogy()
    ax2.set_xlabel('Modes')
    ax2.set_ylabel('Singular values')
    ax2.set_title('PCA, semi log-plot')


    ax3.plot(sigma_energy/np.sum(sigma),'ko',markersize=3)
    ax3.set_xlabel('Modes')
    ax3.set_ylabel('Cumulative energy')
    ax3.set_title('Normalized cumulative energy')
    f.tight_layout()
    f.savefig('PCA_spectrum.png',dpi = 300)

def inverseTransform(u, s, vh, r_max, r_step):
# Perform inverse transform and map from the embedded space to the original space
# Calculate reconstruction error
# Input:
# u, s, vh - outputs of the SVD algorithm
# r_max - max number of modes to keep
# r_step - mode number increment for calculating the reconstruction error
# Output:
# X_reconstruced - reconstructed data with r_max modes
# err_rec - vector containing the reconstruction error for 1 to r_max modes
    err_rec = np.zeros(len(range(1,r_max+1,r_step)))
    i = 0
    for r in range(1,r_max+1,r_step):
        S_r = np.diag(s[0:r])
        U_r = u[:,0:r]
        Vh_r = vh[0:r,:]
        X_reconstructed = U_r @ S_r @ Vh_r
        err_rec[i] = np.linalg.norm(X-X_reconstructed)/np.linalg.norm(X)
        i += 1
    return X_reconstructed, err_rec

def plotAndSaveReconstruction(err_rec, save_rec, X, X_reconstructed, mesh, out_filename, plotToScreen, r_max, r_step, field_name):
# Plot the reconstruction error as a function of the number of modes used
# Input:
# err_rec - vector containing the reconstruction error for 1 to r_max modes
# save_rec - if True: save the original and reconstructed fields in a vtk file
#            !caution - for large data sets these file will be also large!
# X - original data matrix
# X_reconstructed - reconstructed data matrix with r_max modes
# mesh - mesh object containing the location of the mesh points
# out_filename - name and location of the output vtk files for the reconstruction
# plotToScreen - if True: show the reconstruction plot on the display
# r_max - max number of modes used for reconstruction
# r_step - mode number increment for calculating the reconstruction error
# field_name - velocity field name in the original vtk files (just for removing unnecessary data)



    plt.figure(2)
    plt.plot(range(1,r_max+1,r_step),err_rec,linestyle='--', marker='o')
    plt.xlabel('Modes used for reconstruction')
    plt.ylabel('Relative reconstruction error')
    plt.title('PCA reconstruction error')
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('PCA_rec_error.png',dpi = 200)
    if(plotToScreen):
        plt.show()

    if(save_rec):
        meshNew = dsa.WrapDataObject(mesh)
        meshNew.PointData.RemoveArray(field_name) 
        if not transpose_flag:
        # the scikit-learn algorithm by default has a different data arrangement than our definition above
            X_reconstructed = X_reconstructed.T
        for j in range(0,X.shape[1]):
            meshNew.PointData.append(X_reconstructed[:,j], 'reconstructed')
            meshNew.PointData.append(X[:,j], 'original')
            if case == 'turbulent_channel/':
                writer = vtk.vtkRectilinearGridWriter()
            else:
                writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(out_filename + str(j)+ '.vtk')
            writer.SetInputData(meshNew.VTKObject)
            writer.Write()


def create_dir(dir_name):
# Function for creating directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


##########################################################################


input_dir = "../data/"
# Selecting fluid flow test case
# Flow over a cylinder = 'flow_over_cylinder/'
# 2D turbulent channel flow = 'turbulent_channel/'
# ICA brain aneurysm = 'ICA/'
# MCA brain aneurysm = 'MCA/'
case = 'flow_over_cylinder/'

print('Fluid flow test case: ' + case, flush = True) 
# velocity series file name 
filename = 'velocity_'

# name of the velocity field in the vtk files:
if case == 'flow_over_cylinder/':
    field_name = 'f_18'
elif case == 'turbulent_channel/':
    field_name = 'Velocity'
else:
    field_name = 'velocity'

# set vtk reader type and file extensions
if case == 'turbulent_channel/':
    reader = vtk.vtkRectilinearGridReader()
    file_ext = '.vtk'
else:
    reader = vtk.vtkXMLUnstructuredGridReader()
    file_ext = '.vtu'

if case == 'flow_over_cylinder/':
    t_transient = 999
    t_end = 1999
elif case == 'turbulent_channel/':
    t_transient = 0
    t_end = 2000
else:
    t_transient = 0
    t_end = 1000

X, mesh = read_velocity_data(input_dir + case, filename, field_name, reader, file_ext, t_transient, t_end)

# For PCA the spatial and temporal reduction is the same


X = convertToMagnitude(X)

subtract_mean_flag = True

if subtract_mean_flag:
    X, X_mean = subtract_mean(X)

n = X.shape[0]
m = X.shape[1]
print("Data matrix X is n by m:", n, "x", m, flush = True)

# Perform PCA/POD
start = time.time()
u, s, vh = np.linalg.svd(X,full_matrices = False)
end = time.time()
print('Time elapsed for PCA:',end - start, ' s')

# plot singular value spectrum and normalized cumulative energy
plotSpectrum(s)

# save POD modes in a VTK file
out_filename = 'PCA_modes.vtk'
modesToSave = 8
save_modes(u, mesh, out_filename, modesToSave, field_name)

r_max = 10

r_step = 1

X_reconstructed, err_rec = inverseTransform(u, s, vh, r_max, r_step)
# Save reconstruction errors for further post-processing
err_rec_mat = np.stack((range(1,r_max+1,1) ,err_rec),axis = -1)
np.savetxt("error_rec.csv", err_rec_mat, delimiter=",")
save_rec_flag = False

reconstruction_filename = ''

if save_rec_flag == True:
    reconstruction_dir = 'Reconstruction'
    create_dir('Reconstruction')
    reconstruction_filename = reconstruction_dir + '/reconstruction_KPCA_'
    
plotToScreen = True

plotAndSaveReconstruction(err_rec,save_rec, X, X_reconstructed, mesh, reconstruction_filename, plotToScreen, r_max, r_step, field_name)

