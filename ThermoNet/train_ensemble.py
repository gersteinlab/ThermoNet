#!/usr/bin/env python3

# import required modules
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
import numpy as np
from argparse import ArgumentParser


def parse_cmd_args():
    """
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--direct_features', dest='direct_features', type=str, required=True,
           help='Features for the training set.')
    parser.add_argument('-i', '--inverse_features', dest='inverse_features', type=str, required=True,
           help='Features for the training set.')
    parser.add_argument('-y', '--direct_targets', dest='direct_targets', type=str, required=True,
           help='Targets for the trainig set.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, required=True,
           help='Number of training epochs.')
    parser.add_argument('--prefix', dest='prefix', type=str, required=True,
           help='Prefix for the output file to store trained model.')
    args = parser.parse_args()
    return args


def build_model(model_type='regression', conv_layer_sizes=(16, 16, 16), dense_layer_size=16, dropout_rate=0.5):
    """
    """
    # make sure requested model type is valid
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
        
    # instantiate a 3D convnet
    model = models.Sequential()
    model.add(layers.Conv3D(filters=conv_layer_sizes[0], kernel_size=(3, 3, 3), input_shape=(16, 16, 16, 14)))
    model.add(layers.Activation(activation='relu'))
    for c in conv_layer_sizes[1:]:
        model.add(layers.Conv3D(filters=c, kernel_size=(3, 3, 3)))
        model.add(layers.Activation(activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=dense_layer_size, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    
    # the last layer is dependent on model type
    if model_type == 'regression':
        model.add(layers.Dense(units=1))
    else:
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
    
    return model


def main():
    """
    """
    args = parse_cmd_args()

    print('Loading training data ...')

    # load training dataset
    X_direct = np.load(args.direct_features)
    X_direct = np.moveaxis(X_direct, 1, -1)
    X_inverse = np.load(args.inverse_features)
    X_inverse = np.moveaxis(X_inverse, 1, -1)
    y_direct = np.loadtxt(args.direct_targets)
    y_inverse = -y_direct
    
    
    # instantiate the model for display model architecture
    conv_layer_sizes = (16, 24, 32)
    dense_layer_size = 24
    model = build_model('regression', conv_layer_sizes, dense_layer_size)
    model.summary()
    
    # number of individual models of the ensemble
    k = 10
    sample_size = X_direct.shape[0]
    num_validation_samples = sample_size // k
    for i in range(k):
        # create a validation set
        X_val_direct = X_direct[i * num_validation_samples: (i + 1) * num_validation_samples]
        y_val_direct = y_direct[i * num_validation_samples: (i + 1) * num_validation_samples]
        X_val_inverse = X_inverse[i * num_validation_samples: (i + 1) * num_validation_samples]
        y_val_inverse = y_inverse[i * num_validation_samples: (i + 1) * num_validation_samples]
        X_val = np.concatenate((X_val_direct, X_val_inverse))
        y_val = np.concatenate((y_val_direct, y_val_inverse))

        # create the training data from all other partitions
        X_train_direct = np.concatenate(
                [X_direct[:i * num_validation_samples],
                 X_direct[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        X_train_inverse = np.concatenate(
                [X_inverse[:i * num_validation_samples],
                 X_inverse[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        y_train_direct = np.concatenate(
                [y_direct[:i * num_validation_samples],
                 y_direct[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        y_train_inverse = np.concatenate(
                [y_inverse[:i * num_validation_samples],
                 y_inverse[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        X_train = np.concatenate((X_train_direct, X_train_inverse))
        y_train = np.concatenate((y_train_direct, y_train_inverse))

        # shuffle the training set
        indices = np.arange(0, X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # checkpoint
        model_path = args.prefix + '_member_' + str(i + 1) + '.h5'
        checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                save_best_only=True, mode='min')
        # reduce_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
        callbacks_list = [checkpoint, early_stopping]
        
        # build a model
        model = build_model(model_type='regression', conv_layer_sizes=conv_layer_sizes, 
                dense_layer_size=dense_layer_size, dropout_rate=0.5)
        model.compile(loss='mse', optimizer=optimizers.Adam(
            lr=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            amsgrad=False
            ), 
            metrics=['mae']
        )
        
        # fit the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=8, callbacks=callbacks_list, verbose=1)
    

if __name__ == '__main__':
    main()
