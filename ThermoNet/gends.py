#!/usr/bin/env python3

import os
import sys
import utils
import time
from argparse import ArgumentParser
import numpy as np


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str,
                        help='File that contains a list of protein mutations.')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='File to which to write the dataset.')
    parser.add_argument('-p', '--pdb_dir', dest='pdb_dir', type=str,
                        help='Directory where PDB file are stored.')
    parser.add_argument('-r', '--rotations', dest='rotations', type=float, nargs=3,
                        help='Rotation angles in radian around all three axes.')
    parser.add_argument('--boxsize', dest='boxsize', type=int,
                        help='Size of the bounding box around the mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int,
                        help='Size of the voxel.')
    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        help='Whether to overwrite PDBQT files and mutant PDB files.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Whether to print verbose messages from HTMD function calls.')
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='This flag indicates that the given list of mutations are reverse mutations')
    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args


def main():
    """

    Returns
    -------

    """
    args = parse_cmd()

    # calculate rotation angles
    if args.rotations is not None:
        rotations = np.pi * np.array(args.rotations)
    else:
        rotations = None

    dataset = []
    with open(args.input, 'rt') as f:
        # to store computed wt features
        features_wt_all = {}
        for l in f:
            # get mutation identifiers
            pdb_chain, pos, wt, mt = l.strip().split()

            # make sure that the PDB file for wild-type exists
            pdb_dir = os.path.abspath(args.pdb_dir)
            wt_pdb_path = os.path.join(pdb_dir, pdb_chain, pdb_chain + '_relaxed.pdb')
            if not os.path.exists(wt_pdb_path):
                print('PDB file for wild-type does not exist: ' + wt_pdb_path)
                sys.exit(1)

            # calculate wt features
            if pdb_chain + pos not in features_wt_all:
                print('Now computing features for: {0}'.format(wt_pdb_path))
                # HTMD generates channel-first tensors
                features = utils.compute_voxel_features(pos, wt_pdb_path, boxsize=args.boxsize,
                        voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)
                features_wt_all[pdb_chain + pos] = features

            # create a structural model for the mutant
            mt_pdb_path = os.path.join(pdb_dir, pdb_chain, pdb_chain + '_' + wt + \
                    pos + mt + '_relaxed.pdb')
            if not os.path.exists(mt_pdb_path):
                print('PDB file for mutant does not exist: ' + mt_pdb_path)
                print('Running Rosetta to generate a mutant structure ...')
                mt_pdb_path_tmp = utils.build_mt_struct(wt_pdb_path, chain_id, pos, wt, mt, \
                        args.overwrite, os.path.join(pdb_dir, pdb_chain))
                # rename the mutant PDB file to ${pdb_chain}_${variant}_relaxed.pdb
                os.rename(mt_pdb_path_tmp, mt_pdb_path)

            # calculate mutant features
            print('Now computing features for: {0}'.format(mt_pdb_path))
            # HTMD generates channel-first tensors
            features_mt = utils.compute_voxel_features(pos, mt_pdb_path, boxsize=args.boxsize,
                    voxelsize=args.voxelsize, verbose=args.verbose, rotations=rotations)

            # make the property channels the first axis
            # features = features.swapaxes(0, -1)
            # features_mt = features_mt.swapaxes(0, -1)
            features_wt = features_wt_all[pdb_chain + pos]
            features_wt = np.delete(features_wt, obj=6, axis=0)
            features_mt = np.delete(features_mt, obj=6, axis=0)
            if args.reverse:
                # reverse mutation
                features_combined = np.concatenate((features_mt, features_wt), axis=0)
            else:
                # direct mutation
                features_combined = np.concatenate((features_wt, features_mt), axis=0)
            # wild-type synonymous mutation
            # features_combined = np.concatenate((features_wt, features_wt), axis=-1)
            # features_combined = features_mt - features_wt
            # features of reverse mutations
            # features_combined = features_wt - features_mt
            # features_combined = np.swapaxes(features_combined, 0, -1)
            dataset.append(features_combined)

    # convert into NumPy ndarray
    dataset = np.array(dataset)
    print('The generated dataset has shape: {0}'.format(dataset.shape))

    # write dataset to disk file
    print('Now writing dataset to disk file: {0}'.format(args.output))
    # utils.write_5dtensor(dataset, file=args.output, shape=dataset.shape)
    np.save(args.output, dataset)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print('gends.py took', elapsed, 'seconds to generate the dataset.')
