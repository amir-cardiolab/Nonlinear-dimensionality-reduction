# Nonlinear-dimensionality-reduction

This repository contains the python codes and data files for the following paper:
Comparing different nonlinear dimensionality reduction techniques for data-driven unsteady fluid flow modeling
Hunor Csala, Scott T. M. Dawson, Amirhossein Arzani


The dimensionality reduction python codes requires the following packages to be installed before running the codes:

scikit-learn (for the manifold learning methods)
https://scikit-learn.org/stable/install.html

vtk (for handling reader and writing vtk data files)
https://vtk.org/download/
Anaconda installation:
https://anaconda.org/conda-forge/vtk

matplotlib (for visualization only)
https://anaconda.org/conda-forge/matplotlib

pytorch (for the deep learning methods)
https://pytorch.org/get-started/locally/

Python codes:

PCA - Principal Component Analysis (POD)

LLE - Locally Linear Embedding

KPCA - Kernel PCA

LEM - Laplacian Eigenmaps

Isomap - Isometric mapping

AE	- Autoencoder (spatial reduction)

MDAE - Mode-decomposing Autoencoder (spatial reduction)

T_AE - Autoencoder (temporal reduction)