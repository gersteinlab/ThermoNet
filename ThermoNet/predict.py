#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:55:40 2019

@author: bian
"""

# import required modules
from keras.models import load_model
import numpy as np
from argparse import ArgumentParser


def parse_cmd_args():
    """
    """
    argparser = ArgumentParser()
    argparser.add_argument('-x', '--features', dest='features', type=str, required=True,
                           help='Features of the training set.')
    argparser.add_argument('-m', '--model', dest='model', type=str, required=True,
                           help='HDF5 file to which to write leaned model parameters.')
    argparser.add_argument('-o', '--output', dest='output', type=str, required=True,
                           help='Disk file to which to write predictions on the training set.')
    return argparser.parse_args()


def main():
    """
    """
    # parse command line arguments
    args = parse_cmd_args()

    # load training set
    features = np.load(args.features)
    features = np.moveaxis(features, 1, -1)
    # features = features.reshape((args.number, 14, 16, 16, 16))

    # load trained model
    model = load_model(args.model)
    
    # save predictions
    y_pred = model.predict(features)
    np.savetxt(fname=args.output, X=y_pred[:, 0], fmt='%.3f')
    
    
if __name__ == '__main__':
    main()
