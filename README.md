# ThermoNet
ThermoNet is a computational framework for quantitative prediction of the impact of single-point mutations on protein thermodynamic stability. The core algorithm of ThermoNet is an ensemble of deep 3D convolutional neural networks. 

## Requirements
To use ThermoNet, you would need to install the following software and Python libraries:
  * Rosetta 3.10. Rosetta is a multi-purpose macromolecular modeling suite that is used in ThermoNet for creating and refining protein structures. You can get Rosetta from Rosetta Commons: https://www.rosettacommons.org/software
  * HTMD 1.17. HTMD is a Python library for high-throughput molecular modeling and dynamics simulations. It is called in ThermoNet for voxelizing protein structures and parametrizing the resulting 3D voxel grids.
  * Keras with TensorFlow as the backend.

## Installation
ThermoNet ddG prediction is a multi-step protocol. Each step relies on a specific third-party software that needs to be installed first. In the following, we outline the steps to install them.

### Install Rosetta
1. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
2. Download Rosetta 3.10 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
3. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

### Install HTMD 1.17
The free version of HTMD is free to non-commercial users although it does not come with full functionality. But for the purpose of using it with ThermoNet, the free version is sufficient. Install it by following the instructions listed [here](https://software.acellera.com/install-htmd.html).

### Install TensorFlow and Keras
There are many resources out there that one can follow to install TensorFlow and Keras. I found it easiest to install them with the Anaconda Python distribution.
1. Get the Python 3.7 version [Anaconda 2019.10](https://www.anaconda.com/distribution/) for Linux installer. 
2. Follow the instructions [here](https://docs.anaconda.com/anaconda/install/linux/) to install it.
3. Open anaconda-navigator from the comand line. Go to Environments and search for keras and tensorflow, install all the matched libraries.

## Use ThermoNet
As previously mentioned, the ThermoNet ddG method is a multi-step protocol. We outline the steps needed to make ddG predictions of a given variant or a list of variants in the following. Note that ThermoNet requires that a protein structural model is available for the protein from which the variants are derived. It has not been tested on cases where no protein structures are available. 

1. Run the following command to refine the given protein structure `XXXXX.pdb`:
```bash
relax.static.linuxgccrelease -in:file:s XXXXX.pdb -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false
```
where relax.static.linuxgccrelease is the binary executable of the Rosetta FastRelax protocol for relaxing the given protein structure with the Rosetta energy function. This command will generate a PDB file named XXXXX_relaxed.pdb, which will be used in later steps.

2. Run the following command to create a structural model for each of the variants in the given list:
```bash
rosetta_relax.py --rosetta-bin relax.static.linuxgccrelease -l VARIANT_LIST --base-dir /path/to/where/all/XXXXX_relaxed.pdb/is/stored
```
where VARIANT_LIST is a given file in which each line is a given variant in the format `XXXXX POS WT MUT`. For example, `2ocjA 282 R W`.

3. Run the following command to generate tensors (parameterized 3D voxel grids), which will be the input of the 3D deep convolutional neural networks:
```bash
gends.py -i VARIANT_LIST -o test_direct_stacked_16_1 -p /path/to/where/all/XXXXX_relaxed.pdb/is/stored --boxsize 16 --voxelsize 1 --direct
```
This command will generate a file called `test_direct_stacked_16_1.npy` that stores the parameterized 3D voxel grids for each variant in the given list. This file will be used as input to the ensemble of 3D deep convolutional neural networks.

4. Run predict.py:
```bash
for i in `seq 1 10`; do predict.py -x test_direct_stacked_16_1.npy -m models/conv_16_24_32_dense_24_ensemble_member_${i}.h5 -o test_predictions_${i}.txt; done
```

## Example
In the following, we illustrate the steps to generate ddGs for a list of single-point mutations of the p53 tumor suppressor protein. 
1. Change directory to `examples` where you'll find the PDB file `2ocjA.pdb` and a filed named `p53_variants.txt` which contains a list of p53 single-point mutations.

2. Run the Rosetta FastRelax application to relax the PDB structure. The purpose of doing this is to remove potential high energetic frustructions introduced in the PDB structure and to create a good reference state for creating mutant structures.
```bash
relax.static.linuxgccrelease -in:file:s 2ocjA.pdb -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false
```
This step takes about a minute or so depending on the size of the protein and will generate a file named `2ocjA_relaxed.pdb`.

3. Now create a directory called `2ocjA` and move `2ocjA_relaxed.pdb` to this directory.
```bash
mkdir 2ocjA
mv 2ocjA_relaxed.pdb 2ocjA
```

4. Run the following command to create a structural model for each of the variants in the given list.
```bash
rosetta_relax.py --rosetta-bin relax.static.linuxgccrelease -l p53_mutations.txt --base-dir ./
```
This step will create a PDB file for each mutations in the given list. For example, it will create a file named `2ocjA_relaxed_Q104H_relaxed.pdb` for the mutation `2ocjA 104 Q H`.

5. Run the following command to rename all PDB files.
```bash
rename _relaxed_ _ *_relaxed_*.pdb
```
The purpose of renaming the files is to make the file names conform to the requirement of the `gends.py` utility.

6. Generate a dataset containing the tensor for each mutation. Note that you'll need to run this step in the HTMD conda environment.
```bash
gends.py -i p53_variants.txt -o p53_direct_stacked_16_1 -p ./ --boxsize 16 --voxelsize 16 --direct
```

7. Run predict.py to make ddG predictions.
```bash
for i in `seq 1 10`; do predict.py -x p53_direct_stacked_16_1.npy -m ../models/conv_16_24_32_dense_24_ensemble_member_${i}.h5 -o p53_predictions_${i}.txt; done
```
## References
  1. Rosetta references
    * Leaver-Fay, A., Tyka, M., Lewis, S.M., Lange, O.F., Thompson, J., Jacak, R., Kaufman, K., Renfrew, P.D., Smith, C.A., Sheffler, W., et al. (2011). ROSETTA3: an object-oriented software suite for the simulation and design of macromolecules. Methods Enzymol 487, 545-574
    * Tyka, M.D., Keedy, D.A., Andre, I., Dimaio, F., Song, Y., Richardson, D.C., Richardson, J.S., and Baker, D. (2011). Alternate states of proteins revealed by detailed energy landscape mapping. J Mol Biol 405, 607-618
  2. Protein folding, structure prediction, and thermostability
    * Li, B., Fooksa, M., Heinze, S., and Meiler, J. (2018). Finding the needle in the haystack: towards solving the protein-folding problem computationally. Crit Rev Biochem Mol Biol 53, 1-28 
    * Guerois, R., Nielsen, J.E., and Serrano, L. (2002). Predicting changes in the stability of proteins and protein complexes: A study of more than 1000 mutations. Journal of Molecular Biology 320, 369-387
  3. HTMD
    * Doerr, S., Harvey, M.J., Noe, F., and De Fabritiis, G. (2016). HTMD: High-Throughput Molecular Dynamics for Molecular Discovery. Journal of Chemical Theory and Computation 12, 1845-1852
