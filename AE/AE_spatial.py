import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torchvision import datasets, transforms
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


def create_dir(dir_name):
# Function for creating directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

############################################################################
input_dir = "../data/"
# Selecting fluid flow test case
# Flow over a cylinder = 'flow_over_cylinder/'
# 2D turbulent channel flow = 'turbulent_channel/'
# ICA brain aneurysm = 'ICA/'
# MCA brain aneurysm = 'MCA/'
case = 'ICA/'

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

X = convertToMagnitude(X)

subtract_mean_flag = True

if subtract_mean_flag:
    X, X_mean = subtract_mean(X)

#normalize everything to [0,1]
u=1
l=0
X = (X-np.min(X))/(np.max(X)-np.min(X))*(u-l)+l

n = X.shape[0]
m = X.shape[1]
print("Data matrix X is n by m:", n, "x", m,flush = True)

# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
    print(torch.cuda.get_device_name(device))
else: print('Running on CPU')


# Prepare dataset for pyTorch
# default data arrangement in pytorch is different than ours, so we have to transpose X
X_tensor = torch.from_numpy(X.T)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = 128
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
index_list = torch.randperm(len(dataset)).tolist()
shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)


# Define autoencoder network structure
class Autoencoder_Linear(nn.Module):
    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n,8192),
            nn.ReLU(),
            nn.Linear(8192,2048),
            nn.ReLU(),
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,8)
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Linear(8192,n)
        ) 
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# Define loss and optimiziation parameters
model = Autoencoder_Linear().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(),lr = 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3000, gamma = 0.1)
scheduler_active_flag = True

# Start the training loop

num_epochs = 3
outputs = []
loss_list = []
start = time.time()
for epoch in range(num_epochs):
    batch_iter = 0
    loss_tot = 0.0
    for x in data_loader:
        # x is a list originally, so we have to get the first element which is the tensor
        snapshot = x[0].type(torch.FloatTensor).to(device)
        recon = model(snapshot)
        loss = criterion(recon, snapshot)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        if epoch == num_epochs-1:
            outputs.append((epoch+1,batch_iter, snapshot, recon.detach()))
        batch_iter += 1
    loss_tot = loss_tot/batch_iter
    loss_list.append((epoch, loss_tot))
    print(f'Epoch: {epoch+1}, Total avg loss: {loss_tot:.10f}', flush = True)
    if (scheduler_active_flag):
        scheduler.step()


end = time.time()
print('Time elapsed for AE:',end - start)

# Organize results for saving and visualization
# Unshuffle results and reconstructions
outx_shuffled = []
outxrec_shuffled = []
for i in range(int(np.ceil(m/batchsize))):
    outx_shuffled.append(outputs[i][2])
    outxrec_shuffled.append(outputs[i][3])
x_out_shuffled = torch.cat(outx_shuffled).detach().cpu().numpy()
xrec_out_shuffled = torch.cat(outxrec_shuffled).detach().cpu().numpy()

x_out = np.zeros(x_out_shuffled.shape)
xrec_out = np.zeros(xrec_out_shuffled.shape)

j = 0
for i in index_list:
    x_out[i,:] = x_out_shuffled[j,:]
    xrec_out[i,:] = xrec_out_shuffled[j,:]
    j +=1

error_rec = np.linalg.norm(x_out-xrec_out)/np.linalg.norm(x_out)
print('Relative reconstruction error: %.5e' % (error_rec))

# Save the modes and the reconstructed field
save_rec_flag = True

reconstruction_filename = ''

if save_rec_flag == True:
    reconstruction_dir = 'Reconstruction'
    create_dir('Reconstruction')
    reconstruction_filename = reconstruction_dir + '/reconstruction_AE_'

if(save_rec_flag):
    print('Saving the reconstructed velocity field to ',reconstruction_filename)
    meshNew = dsa.WrapDataObject(mesh)
    meshNew.PointData.RemoveArray(field_name)
    for j in range(0,x_out.shape[0]):
        meshNew.PointData.append(xrec_out[j,:], 'reconstructed')
        meshNew.PointData.append(x_out[j,:], 'original')
        if case == 'turbulent_channel/':
            writer = vtk.vtkRectilinearGridWriter()
        else:
            writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(reconstruction_filename + str(j)+ '.vtk')
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()

# Plot loss as a function of the number of epochs
loss_mat = np.asarray(loss_list)
plt.figure(1)
plt.plot(loss_mat[:,0],loss_mat[:,1],linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AE Loss')
plt.semilogy()
plt.tight_layout()
plt.savefig('AE_loss.png',dpi = 200)
