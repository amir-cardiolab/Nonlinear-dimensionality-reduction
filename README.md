# Nonlinear-dimensionality-reduction

This repository contains the python codes and data files for the following paper:
Comparing different nonlinear dimensionality reduction techniques for data-driven unsteady fluid flow modeling
Hunor Csala, Scott T. M. Dawson, Amirhossein Arzani

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Python codes

Python codes are provided for the different examples presented in the paper. Each dimensionality reduction method can be used with all 4 four data sets (flow over a cylinder, turbulent channel flow, ICA and MCA aneurysms). The test case choice can be selected. The manifold learning codes (LLE, KPCA, LEM, Isomap) can be used for both spatial and temporal reduction by changing a boolean variable, for the autoencoders separate codes are provided for spatial (AE and MDAE) and temporal reduction (T_AE). 
By default PCA and the manifold learning cases are set up for spatial reduction of the flow over a cylinder case with the corresponding hyperparameters (these can be found in Table A.1 in the paper). The autoencoder cases are set up for the ICA aneurysm case with a latent space size of 8.

PCA - Principal Component Analysis (POD)

LLE - Locally Linear Embedding

KPCA - Kernel PCA

LEM - Laplacian Eigenmaps

Isomap - Isometric mapping

AE	- Autoencoder (spatial reduction)

MDAE - Mode-decomposing Autoencoder (spatial reduction)

T_AE - Autoencoder (temporal reduction)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Installation:
The dimensionality reduction python codes requires the following packages to be installed before running the codes:

* scikit-learn (for the manifold learning methods - LLE, KPCA, LEM, Isomap)
https://scikit-learn.org/stable/install.html

* vtk (for handling vtk input and output data files)
https://vtk.org/download/
    * Anaconda installation: https://anaconda.org/conda-forge/vtk

* matplotlib (for visualization only)
https://anaconda.org/conda-forge/matplotlib

* pytorch (for the deep learning methods - AE, MDAE, T_AE)
https://pytorch.org/get-started/locally/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data:

The datasets used for test cases are too large for Github, therefore they are provided with the following link:
https://uofu.box.com/s/z4qyg0fbnsdk9j8bultpyy9hxcawewkp
They should be placed in the */data/* folder under the corresponding test case name. There is a zipped version of all four test cases, but the cases are also available for download separately.