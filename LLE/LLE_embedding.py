import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
from sklearn.manifold import LocallyLinearEmbedding
import time
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
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



def performLLE(X, k_NN, r, reg, transpose_flag):
# Perform Locally Linear Embedding with scikit-learn
# Input:
# X - data matrix
# k_NN - number of nearest neighbors
# r - number of components to keep in the latent space
#   - Note: runtime is highly dependent on this
# reg - regularization parameter
# transpose_flag - temporal vs spatial arrangement of data
#        if True: temporal size of the data will be reduced (leading to spatial modes)
#        if False: spatial size of the data will be reduced (NOT leading to spatial modes)    
# Output:
# X_lle - LocallyLinearEmbeding object
#       - X_lle.embedding_ contains the embedded vectors
    start = time.time()
    embedding = LocallyLinearEmbedding(n_components = r, n_neighbors = k_NN, reg = reg, random_state = 42)
    if not transpose_flag:
    # the scikit-learn algorithm by default has a different data arrangement than our definition above
    # By default it will reduce the size of the rows (temporal embedding), while our default is to reduce the size of the columns (spatial reduction)
    # !!Note that if the tranpose_flag = True, i.e, we are doing temporal reduction, the RAM requirement is very high!!! (Can be up to 64 GB)
    # Temporal reduction (transpose_flag = True) needs a much longer computational time than spatial reduction (transpose_flag = False)
        X = X.T
    X_lle = embedding.fit(X)
    end = time.time()

    print('Time elapsed for LLE embedding:',end - start, ' s', flush = True)
    print('Shape after embedding:', X_lle.embedding_.shape, flush = True)

    return X_lle


def save_modes(X_lle, mesh, output_file, modesToSave, field_name):
# Save the spatial modes to a VTK file for visualization
# Input:
# X_lle - LLE object
# mesh - mesh object containing the location of the mesh points
# output_file - name and location of the output vtk file for the modes
# modesToSave - number of modes to save to the VTK file
# field_name - velocity field name in the original vtk files (just for removing unnecessary data)


    mesh = dsa.WrapDataObject(mesh)

    if modesToSave > X_lle.embedding_.shape[1]:
        modesToSave = X_lle.embedding_.shape[1]
        print('Max number of modes is', X_lle.embedding_.shape[1])
    
    print('Saving the first',modesToSave, 'LLE modes to ',out_filename, flush = True)

    for i in range(0,modesToSave,1):
        U_i = X_lle.embedding_[:,i]
        meshNew = dsa.WrapDataObject(mesh)
        meshNew.PointData.RemoveArray(field_name)
        mesh.PointData.append(U_i, 'mode_'+str(i))
        if i == 0 and subtract_mean_flag:           #save the temporal mean
            mesh.PointData.append(X_mean, 'U_mean')
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(mesh.VTKObject)
        writer.Write()


def inverseTransform(X, X_lle, r_max, r_step, k_NN, reg, transpose_flag):
# Mapping back to the original space from the latent space and calculating reconstruction errors
# Inputs:
# X - original data matrix
# X_lle - LLE object
# r_max - max number of components used for reconstruction
# r_step - mode number increment for calculating the reconstruction error
# k_NN - number of nearest neighbors used for LLE (same as for the performLLE function)
# reg - regularization parameter (same as for the performLLE function)
# Output:
# X_reconstructed - reconstructed data matrix with r_max modes
# err_rec - vector containing the reconstruction error for 1 to r_max modes
    err_rec = np.zeros(len(range(1,r_max+1,r_step)))
    if not transpose_flag:
    # the scikit-learn algorithm by default has a different data arrangement than our definition above
        X = X.T
    i = 0
    for r in range(1,r_max+1,r_step):
        # get the first r components of the embedded vectors
        Y = X_lle.embedding_[:,0:r]

        # calculate the weights for expressing each point in the embedded space as a linear combination of its neighbors
        W = barycenter_kneighbors_graph(Y, n_neighbors = k_NN, reg = reg)
        # reconstruct the data points in the high dimensional space from its neighbors using the weights calculated based on the embedded space
        X_reconstructed = W @ X

        # calculate reconstruction error over all snapshots using Frobenius norm
        err_rec[i] = np.linalg.norm(X-X_reconstructed)/np.linalg.norm(X)
        i +=1
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



    plt.figure(1)
    plt.plot(range(1,r_max+1,r_step),err_rec,linestyle='--', marker='o')
    plt.xlabel('Modes used for reconstruction')
    plt.ylabel('Relative reconstruction error')
    plt.title('LLE reconstruction error')
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('LLE_rec_error.png',dpi = 200)
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
#case = 'turbulent_channel/'
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

# transpose_flag - temporal vs spatial arrangement of data
#        if True: temporal size of the data will be reduced (leading to spatial modes)
#        if False: spatial size of the data will be reduced (NOT leading to spatial modes) 
transpose_flag = True

# convertToMagnitude_flag 
#                   if True: velocity magnitude will be used
#                   if False: velocity vector will be used
convertToMagnitude_flag = True


X = convertToMagnitude(X)

subtract_mean_flag = True

if subtract_mean_flag:
    X, X_mean = subtract_mean(X)

n = X.shape[0]
m = X.shape[1]
print("Data matrix X is n by m:", n, "x", m, flush = True)

#Locally Linear Embedding w/ scikit learn

# k_NN - number of nearest neighbors used for LLE
# r_max - number of components to be kept in the latent space
# reg - regularization parameter
k_NN = 10
r_max = 10
reg = 1e-9

X_lle = performLLE(X, k_NN, r_max, reg, transpose_flag)


if transpose_flag:
    out_filename = 'LLE_modes.vtk'
    modesToSave = 8
    save_modes(X_lle, mesh, out_filename, modesToSave, field_name)

# Reconstruction
# r_step - mode number increment for calculating the reconstruction error
r_step = 1

X_reconstructed, err_rec = inverseTransform(X,X_lle, r_max, r_step, k_NN, reg, transpose_flag)

# Save reconstruction errors for further post-processing
err_rec_mat = np.stack((range(1,r_max+1,r_step) ,err_rec),axis = -1)
np.savetxt("error_rec.csv", err_rec_mat, delimiter=",")

save_rec_flag = False

reconstruction_filename = ''

if save_rec_flag == True:
    reconstruction_dir = 'Reconstruction'
    create_dir('Reconstruction')
    reconstruction_filename = reconstruction_dir + '/reconstruction_LLE_'

plotToScreen = True

plotAndSaveReconstruction(err_rec,save_rec_flag, X, X_reconstructed, mesh, reconstruction_filename, plotToScreen, r_max, r_step, field_name)



