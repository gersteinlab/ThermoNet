#!/usr/bin/env python3

from htmd.ui import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, getCenters
from argparse import ArgumentParser


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-o', '--output-prefix', dest='output_prefix', type=str,
                        help='Prefix prepended to the output file.')
    parser.add_argument('-s', '--mutation-site', dest='mutation_site', type=str,
                        help='Prefix prepended to the output file.')
    parser.add_argument('-p', '--pdb-file', dest='pdb_file', type=str,
                        help='PDB file of the protein structure/model.')
    parser.add_argument('-r', '--rotations', dest='rotations', type=float, nargs=3,
                        help='Rotation angles in radian around all three axes.')
    parser.add_argument('--boxsize', dest='boxsize', type=int, default=16,
                        help='Size of the bounding box around the mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int, default=1,
                        help='Size of the voxel.')
    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args


def compute_voxels(mutation_site, pdb_file, verbose=False,
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
    if center.size == 0:
        center = mol.get('coords', 'resid ' + str(mutation_site) + ' and name CA')
    #
    voxel_centers = getCenters(prot, boxsize=[boxsize, boxsize, boxsize],
                center=center.flatten(), voxelsize=voxelsize)
    if rotations is None:
        voxels, _, _ = getVoxelDescriptors(prot, center=center.flatten(),
                boxsize=[boxsize, boxsize, boxsize], voxelsize=voxelsize, validitychecks=False)
    else:
        rotated_voxel_centers = rotateCoordinates(voxel_centers[0], rotations, center.flatten())
        voxels, _ = getVoxelDescriptors(prot, usercenters=rotated_voxel_centers, validitychecks=False)
    # return features
    nchannels = voxels.shape[1]
    n_voxels = int(boxsize / voxelsize)
    voxels = voxels.transpose().reshape((nchannels, n_voxels, n_voxels, n_voxels))

    return voxels, voxel_centers[0]


def main():
    """
    """
    args = parse_cmd()

    voxels, voxel_centers = compute_voxels(args.mutation_site, args.pdb_file, 
            boxsize=args.boxsize, voxelsize=args.voxelsize)

    for i, channel in enumerate(voxels):
        values = channel.flatten()
        pdb_records = []
        if len(voxel_centers) != len(values):
            raise ValuueError('coords and values have different lengths')
        else:
            j = 1
            for c, v in zip(voxel_centers, values):
                record = '{0: <6}{1: >5}  {2}  {3} {4}{5: >4}    {6: >8.3f}{7: >8.3f}{8: >8.3f}{9: >6.2f}{10: >6.2f}           C'.\
                        format('ATOM', j, 'CA', 'ALA', 'A', j, c[0], c[1], c[2], 1.00, v)
                pdb_records.append(record)
                j = j + 1

        # write to a PDB file
        voxel_pdb_file = args.output_prefix + '_channel_' + str(i+1) + '.pdb'
        with open(voxel_pdb_file, 'wt') as opf:
            opf.write('\n'.join(pdb_records))


if __name__ == '__main__':
    main()
