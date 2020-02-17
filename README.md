# ThermoNet
ThermoNet is a computational framework for quantitative prediction of the impact of single-point mutations on protein thermodynamic stability. The core algorithm of ThermoNet is an ensemble of deep 3D convolutional neural networks. 

## Requirements
To use ThermoNet, you would need to install the following software and Python libraries:
  * Rosetta 3.10. Rosetta is a multi-purpose macromolecular modeling suite that is used in ThermoNet for creating and refining protein structures. You can get Rosetta from Rosetta Commons: https://www.rosettacommons.org/software
  * HTMD 1.17. HTMD is Python library for high-throughput molecular modeling and dynamics simulations. It is called in ThermoNet for voxelizing protein structures and parametrizing the resulting 3D voxel grids.
  * Keras with TensorFlow as the backend.

## Installation
ThermoNet ddG prediction is a multi-step protocol. Each step replies on a specific third-party software that needs to be installed first. In the following, we outline the steps to installing them.

### Install Rosetta
1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.10 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

### Install HTMD 1.17
The free version of HTMD is free to non-commercial users although it does not come with full functionality. But for the purpose of using it with ThermoNet, the free version is sufficient. Install it by following the instructions listed [here](https://software.acellera.com/install-htmd.html).

### Install TensorFlow and Keras
There are many resources out there that one can follow to install TensorFlow and Keras. I found it easiest to install them with Anaconda  Python distribution.
1. Get the Python 3.7 version [Anaconda 2019.10](https://www.anaconda.com/distribution/) for Linux installer. 
2. Follow the instructions [here](https://docs.anaconda.com/anaconda/install/linux/) to install it.
3. Open anaconda-navigator from the comand line. Go to Environments and search for keras and tensorflow, install all the matched libraries.
