#!/usr/bin/env python3

import os
import sys
import numpy as np
from htmd.ui import *
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors,getCenters, rotateCoordinates
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from Bio.PDB import PDBParser, NeighborSearch
import subprocess
import math


ADT_PATH = '/usr/local/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/'
ROSETTA_BIN = '/ysm-gpfs/apps/software/Rosetta/3.10/main/source/bin/relax.static.linuxgccrelease'


def vector_add(v, w):
    """
    Adding two vectors in a element-wise manner.

    Parameters
    ----------
    v : list
        Vector v.
    w : list
        Vector w.

    Returns
    -------
    list
        The result of adding vectors v and w.

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """
    Subtracting vector w from vector v in a element-wise manner.

    Parameters
    ----------
    v : list
        Vector v.
    w : list
        Vector w.

    Returns
    -------
    list
        The result of subtracting vector w from vector v.

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """Compute a vector whose ith element is the sum of the ith elements of
    the input vectors.

    Parameters
    ----------
    vectors

    Returns
    -------
    list
        A list whose ith element is the sum of the ith elements of the input
        vectors.

    """
    result = vectors[0]
    for v in vectors[1:]:
        result = [a + b for a, b in zip(result, v)]
    return result


def scalar_multiply(c, v):
    """
    Multiply each element of vector v by the scalar c.

    Parameters
    ----------
    c : float
        A number.
    v : list
        A list.

    Returns
    -------
    list
        A list whose ith element is the ith element of the input vector
        multiplied by c.

    """
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    """Compute the vector whose ith element is the mean of the ith elements
    of the input vectors.

    Parameters
    ----------
    vectors

    Returns
    -------
    list
        A list whose ith element is the mean of the ith elements of the
        input vectors.

    """
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v, w):
    """
    Computes the dot product of vectors v and w.

    Parameters
    ----------
    v : list
        Vector v.
    w : list
        Vector w.

    Returns
    -------
    float
        The dot product of vectors v and w.

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    """Computes the sum of squares of each element in the input vector.

    Parameters
    ----------
    v : list
        A list.

    Returns
    -------
    float
        The sum of squares of each element in the input vector.

    """
    return dot(v, v)


def squared_distance(v, w):
    """
    The square of the Euclidean distance between v and w.

    Parameters
    ----------
    v : list
        Vector v.
    w : list
        Vector w.

    Returns
    -------
    float
        The square of the Euclidean distance between v and w.

    """
    return sum_of_squares(vector_subtract(v, w))


def distance(v, w):
    """
    Computes the Euclidean distance between v and w.

    Parameters
    ----------
    v : list
        Vector v.
    w : list
        Vector w.

    Returns
    -------
    float
        The Euclidean distance between v and w.

    """
    return math.sqrt(squared_distance(v, w))


def naive_relu(x):
    """
    A naive implementation of the ReLU activation function.

    Parameters
    ----------
    x : Numpy NDArray

    Returns
    -------
    NumPy NDArray
        Input matrix after filtered by the ReLU activation function.

    """
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def native_add(x, y):
    """

    Parameters
    ----------
    x : NumPy 2DArray
    y : NumPy 2DArray

    Returns
    -------
    NumPy 2DArray

    """
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            x[i, j] += y[i, j]

    return x


def compute_interaction_center(pdb_file, mutation_site):
    """Computes the geometric center of all heavy atoms interacting with the mutated residue.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB file containing the structure of the protein.
        
    mutation_size : int
        An integer designating the residue sequence ID.
        
    Returns
    -------
    NumPy ndarray
        The Cartesian coordinates of the geometric center
    """
    pdb_parser = PDBParser(PERMISSIVE=1)
    model = pdb_parser.get_structure(id='tmp', file=pdb_file)
    
    # get all heavy atoms of the protein
    all_heavy_atoms = [a for a in model.get_atoms() if a.element != 'H']
    
    # get the mutated residue
    mutation_res = None
    for res in model.get_residues():
        if res.get_id()[1] == int(mutation_site):
            mutation_res = res
            break
    
    # get sidechain atoms
    if mutation_res is not None:
        heavy_atoms = [a for a in mutation_res.get_list() if a.element != 'H']
        # if only four heavy atoms, aka, GLY, use CA
        if len(heavy_atoms) == 4:
            side_chain_atoms = [heavy_atoms[1]]
        else:
            side_chain_atoms = mutation_res.get_list()[4:]
    else:
        print('Invalid mutation site: {}'.format(mutation_site))
        sys.exit(1)
        
    # search for neighbnoring atoms
    ns = NeighborSearch(atom_list=all_heavy_atoms, bucket_size=10)
    all_interaction_atoms = []
    for a in side_chain_atoms:
        interaction_atoms = ns.search(center=a.coord, radius=5, level='A')
        all_interaction_atoms += interaction_atoms
    # remove duplicates
    all_interaction_atoms = set(all_interaction_atoms)
    
    # compute geometric center of all interaction atoms
    geometric_center = np.zeros((3,))
    for a in all_interaction_atoms:
        geometric_center += a.coord / len(all_interaction_atoms)

    return geometric_center


def compute_pdbqt(pdb_file, overwrite=False):
    """Assign AutoDock4 atom types and Gasteiger partial charges to all atoms of the receptor.

    Also build bonds and hydrogens.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file containing the structure of the receptor.

    Returns
    -------
    This function generates the AutoDock4 PDBQT file for the input receptor. Result is written to disk
    files, there are no explicit return values.

    """
    dirname = os.path.dirname(pdb_file)
    prefix = os.path.basename(pdb_file).split('.')[0]

    # if the PDBQT file already exists and not to overwrite, return its path
    pdbqt_path = dirname + '/' + prefix + '.pdbqt'
    if os.path.exists(pdbqt_path) and not overwrite:
        return pdbqt_path

    # construct AutoDock command and run it
    adt_cmd = '{}/prepare_receptor4.py -r {} -A bonds_hydrogens -U ' \
              'deleteAltB -o {}'.format(ADT_PATH, pdb_file, dirname + '/' + prefix + '.pdbqt')
    print('Prepare receptor: {}'.format(adt_cmd))
    response = subprocess.run(adt_cmd, shell=True)

    # if AutoDock command is successful, return the path to the PDBQT file
    if response.returncode == 0:
        return pdbqt_path
    else:
        print('AutoDock Tools command: {} failed!'.format(adt_cmd))
        sys.exit(1)


def compute_voxel_features(mutation_site, pdb_file, verbose=False, 
        boxsize=16, voxelsize=1, rotations=None):
    """Compute voxel features around the mutation site.

    Parameters
    ----------
    pdbqt_wt : AutoDock PDBQT file
        AutoDock PDBQT file of the wild-type protein.
    pdbqt_mt : AutoDock PDBQT file
        AutoDock PDBQT file of the mutant protein.
    mutation_site : int
        Residue sequence position where the mutation took place.
    rotations : list
        Rotation angles in radian around x, y, and z axes.

    Returns
    -------
    NumPy nd-array

    """
    mol = Molecule(pdb_file)
    prot = prepareProteinForAtomtyping(mol, verbose=verbose)
    center = mol.get('coords', 'resid ' + str(mutation_site) + ' and name CB')
    # center_wt = compute_interaction_center(pdb_file_wt, mutation_site)
    if center.size == 0:
        center = mol.get('coords', 'resid ' + str(mutation_site) + ' and name CA')
    #
    if rotations is None:
        features, _, _ = getVoxelDescriptors(prot, center=center.flatten(), 
                boxsize=[boxsize, boxsize, boxsize], voxelsize=voxelsize, validitychecks=False)
    else:
        voxel_centers = getCenters(prot, boxsize=[boxsize, boxsize, boxsize], 
                center=center.flatten(), voxelsize=voxelsize)
        rotated_voxel_centers = rotateCoordinates(voxel_centers[0], rotations, center.flatten())
        features, _ = getVoxelDescriptors(prot, usercenters=rotated_voxel_centers, validitychecks=False)
    # return features
    nchannels = features.shape[1]
    n_voxels = int(boxsize / voxelsize)
    features = features.transpose().reshape((nchannels, n_voxels, n_voxels, n_voxels))

    return features


def build_mt_struct(pdb_wt, chain_id, pos, wt, mt, overwrite=False, outdir=None):
    """Build a structural model for the given mutant protein.

    Parameters
    ----------
    pdb_wt : str
        Path to the PDB file that contains the structure of the wild-type protein.
    chain_id : char
        ID of the chain to which the mutant is to be introduced.
    pos : str
        Sequence position where the mutant is to be introduced.
    wt : str
        Amino acid one-letter of the wild-type residue.
    mt : char
        Amino acid one-letter code of the mutant to be introduced.
    outdir : str
        Directory to which to write all the output files.

    Returns
    -------
    str
        Path to the PDB file of the mutant structural model. The structural
        model of the mutant protein is written to a disk file, there will
        be no explicit return value if the Rosetta subprocess fails.
    """
    # make sure that output directory is accessible
    if outdir is None:
        outdir = 'output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # if the mutant model already exists and not to overwrite, return its path
    pdb_wt_basename = os.path.basename(pdb_wt)
    mt_path = os.path.abspath(outdir) + '/' + pdb_wt_basename.split('.')[0] \
            + '_' + wt + pos + mt + '_relaxed.pdb'
    if os.path.exists(mt_path) and not overwrite:
        return mt_path

    # create a temporary resfile for Rosetta
    res_file = '/tmp/resfile.txt'
    with open(res_file, 'wt') as f:
        f.write('NATAA\nSTART\n\n')
        f.write('{} {} PIKAA {}'.format(pos, chain_id, mt))

    # construct Rosetta command and run it
    rosetta_cmd = '{} -in:file:s {} -relax:constrain_relax_to_start_coords '\
            '-relax:respect_resfile -packing:resfile  {} -out:nstruct 1 '\
            '-ignore_unrecognized_res -out:suffix _{} -out:path:all {} '\
            ' -overwrite -out:no_nstruct_label'.format(
                ROSETTA_BIN, pdb_wt, res_file, wt + pos + mt + '_relaxed', outdir
            )
    response = subprocess.run(rosetta_cmd, shell=True)

    # return the path to the output PDB file if Rosetta run is successful
    if response.returncode == 0:
        return mt_path
    else:
        print('Rosetta command: {} failed'.format(rosetta_cmd))
        sys.exit(1)


def write_5dtensor_txt(tensor, file, shape):
    """

    Parameters
    ----------
    tensor
    file
    shape

    Returns
    -------

    """
    with open(file, 'wt') as outfile:
        outfile.write('# Dataset shape: {0}\n'.format(shape))
        for sample in tensor:
            outfile.write('# New sample\n')
            for i in range(sample.shape[0]):
                outfile.write('# New channel\n')
                channel = sample[i, :, :, :]
                for new_slice in channel:
                    outfile.write('# New slice\n')
                    np.savetxt(outfile, new_slice, fmt='%-7.2f')


def main():
    """

    Returns
    -------

    """
    pass
