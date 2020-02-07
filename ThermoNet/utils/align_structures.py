#!/usr/bin/env python3
"""
Created on Fri May 31 12:07:44 2019

@author: Bian Li
@email: bian.li@yale.edu
"""

from pymol import cmd
from argparse import ArgumentParser


def parse_cmd_args():
    """Set up a command line argument parser
    
    Returns
    -------
    A set of command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-m', '--mobile', dest='mobile', required=True, type=str, 
                        help='PDB file of the structure to be aligned.')
    parser.add_argument('-t', '--target', dest='target', required=True, type=str,
                        help='PDB file of the structure to which to align to.')
    parser.add_argument('-o', '--output', dest='output', required=True, type=str,
                        help='PDB file to which to write the aligned structure.')
    return parser.parse_args()


def main():
    """This is the main function that does structural alignment.
    """
    # parse command line arguments
    args = parse_cmd_args()
    
    # load structures
    cmd.load(args.mobile, 'mobile')
    cmd.load(args.target, 'target')
    
    # align the mobile structure to the target
    results = cmd.align('mobile', 'target')
    print('%-4s\t%-9s\t%-12s' % ('RMSD', 'No. Atoms', 'No. Residues'))
    print('%-4.2f\t%-9d\t%-12d' % (results[0], results[1], results[-1]))
    
    # save the aligned mobile structure
    cmd.save(args.output, 'mobile')
    

if __name__ == '__main__':
    main()