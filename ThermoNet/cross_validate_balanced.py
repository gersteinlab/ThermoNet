#!/usr/bin/env python3

# import required modules
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.models import load_model
from scipy import stats
import numpy as np
from argparse import ArgumentParser
import os


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
    # parser.add_argument('-n', '--examples', dest='examples', type=int, required=True,
    #       help='Number of training examples.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, required=True,
           help='Number of training epochs.')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, required=False,
           help='Output file to store cross-validation results.')
    args = parser.parse_args()
    return args


def build_model(model_type='regression', conv_layer_sizes=(16, 16, 16), dense_layer_sizes=(16, 16), dropout_rate=0.5):
    """
    """
    # make sure requested model type is valid
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
        
    # instantiate a 3D convnet
    model = models.Sequential()
    model.add(layers.Conv3D(filters=conv_layer_sizes[0], kernel_size=(3, 3, 3), input_shape=(8, 8, 8, 14)))
    model.add(layers.Activation(activation='relu'))
    for x in conv_layer_sizes[1:]:
        model.add(layers.Conv3D(filters=x, kernel_size=(3, 3, 3)))
        model.add(layers.Activation(activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=dense_layer_sizes[0], activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    if len(dense_layer_sizes) == 2:
        model.add(layers.Dense(units=dense_layer_sizes[1], activation='relu'))
        model.add(layers.Dropout(rate=dropout_rate))
    
    # the last layer is dependent on model type
    if model_type == 'regression':
        model.add(layers.Dense(units=1))
    else:
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
    
    return model


def cross_validate(X_direct, X_inverse, y_direct, y_inverse, conv_layer_sizes=(8, 8, 8), dense_layer_sizes=(8, 8), k=5, epochs=100):
    """
    """
    assert len(X_direct) == len(X_inverse) and len(y_direct) == len(y_inverse) and len(X_direct) == len(y_direct)
    
    print('===============================================================')
    print('Cross-validating convolutional architecture: ', conv_layer_sizes)
    print('===============================================================')
    
    num_validation_samples = len(X_direct) // k
    all_histories = []
    all_predictions = []
    
    # shuffle the data set
    rand_idx = np.arange(0, len(X_direct))
    np.random.shuffle(rand_idx)
    X_direct = X_direct[rand_idx]
    X_inverse = X_inverse[rand_idx]
    y_direct = y_direct[rand_idx]
    y_inverse = y_inverse[rand_idx]
    np.savetxt('/tmp/rand_idx.txt', rand_idx, fmt='%d')
    
    # cross validate
    for i in range(k):
        print('-------------------------')
        print('Processing fold #', i + 1)
        print('-------------------------')
        
        # prepares the validation data from partition # i
        val_X_direct = X_direct[i * num_validation_samples: (i + 1) * num_validation_samples]
        val_y_direct = y_direct[i * num_validation_samples: (i + 1) * num_validation_samples]
        val_X_inverse = X_inverse[i * num_validation_samples: (i + 1) * num_validation_samples]
        val_y_inverse = y_inverse[i * num_validation_samples: (i + 1) * num_validation_samples]
        val_X = np.concatenate((val_X_direct, val_X_inverse))
        val_y = np.concatenate((val_y_direct, val_y_inverse))
        
        # prepares the training data from all other partitions
        train_X_direct = np.concatenate(
                [X_direct[:i * num_validation_samples],
                 X_direct[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        train_X_inverse = np.concatenate(
                [X_inverse[:i * num_validation_samples],
                 X_inverse[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        train_y_direct = np.concatenate(
                [y_direct[:i * num_validation_samples],
                 y_direct[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        train_y_inverse = np.concatenate(
                [y_inverse[:i * num_validation_samples],
                 y_inverse[(i + 1) * num_validation_samples:]],
                 axis=0
                )
        train_X = np.concatenate((train_X_direct, train_X_inverse))
        train_y = np.concatenate((train_y_direct, train_y_inverse))

        # checkpoint
        filepath='/ysm-gpfs/pi/gerstein/bl626/projects/thermonet/results/models/' + 'fold' + str(i + 1) + '_best_model.hdf5'
        # checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
        #         save_best_only=True, mode='min')
        # callbacks_list = [checkpoint]
        
        # build a model
        model = build_model(model_type='regression', conv_layer_sizes=conv_layer_sizes, 
                dense_layer_sizes=dense_layer_sizes, dropout_rate=0.5)
        model.compile(loss='mse', optimizer=optimizers.Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False
            ), 
            metrics=['mae']
        )
        
        # fit the model
        history = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=epochs,
                batch_size=8)
        
        # evaluate the model on the valitaion set
        all_histories.append(history)
        
        # load the best model for predictions
        # best_model = load_model(filepath)
        # predictions = best_model.predict(val_X)
        # all_predictions.append(np.column_stack((predictions[:, 0], val_y)))

    # get mean validation loss and mean validation mean absolute error
    mean_val_losses = np.mean(np.column_stack([x.history['val_loss'] for x in all_histories]), axis=1)
    mean_val_maes = np.mean(np.column_stack([x.history['val_mean_absolute_error'] for x in all_histories]), axis=1)

    # return mean_val_losses, mean_val_maes, all_predictions
    return mean_val_losses, mean_val_maes


def main():
    """
    """
    args = parse_cmd_args()

    print('Loading training data ...')

    # load training dataset
    X_direct = np.load(args.direct_features)
    # by default, TensorFlow takes channel-last tensors
    X_direct = np.moveaxis(X_direct, 1, -1)
    # X_direct = X_direct.reshape((args.examples, 7, 16, 16, 16))
    X_inverse = np.load(args.inverse_features)
    X_inverse = np.moveaxis(X_inverse, 1, -1)
    # X_inverse = X_inverse.reshape((args.examples, 7, 16, 16, 16))
    y_direct = np.loadtxt(fname=args.direct_targets)
    y_inverse = -y_direct
    # X_direct_dummy = np.load('/ysm-gpfs/pi/gerstein/bl626/projects/thermonet/data/datasets/strum_dummy_variants_popmusic_removed_direct_16_1.npy')
    # X_direct = X_direct.reshape((args.examples, 7, 16, 16, 16))
    # X_inverse_dummy = np.load('/ysm-gpfs/pi/gerstein/bl626/projects/thermonet/data/datasets/strum_dummy_variants_popmusic_removed_inverse_16_1.npy')

    # instantiate the model
    conv_layer_sizes = [(16, 24, 32)]
    dense_layer_sizes = [(24,)]

    for conv_layer_size in conv_layer_sizes:
        for dense_layer_size in dense_layer_sizes:
            model = build_model('regression', conv_layer_size, dense_layer_size)
            model.summary()
            # five-fold cross-validation
            mean_val_losses, mean_val_maes = cross_validate(X_direct, X_inverse, y_direct, y_inverse, conv_layer_size, dense_layer_size, k=5, epochs=args.epochs)

            np.savetxt(os.path.join(args.outdir, 'conv_' + '_'.join(str(x) for x in conv_layer_size) + 
                '_dense_' + '_'.join(str(y) for y in dense_layer_size) + '_mean_val_losses.txt'), mean_val_losses, fmt='%.3f')
            np.savetxt(os.path.join(args.outdir, 'conv_' + '_'.join([str(x) for x in conv_layer_size]) + 
                '_dense_' + '_'.join(str(y) for y in dense_layer_size) + '_mean_val_maes.txt'), mean_val_maes, fmt='%.3f')
            # for i, p in enumerate(all_predictions):
            #     np.savetxt(os.path.join(args.outdir, 'conv_' + '_'.join(str(x) for x in conv_layer_size) + 
            #         '_dense_' + '_'.join(str(y) for y in dense_layer_size) + '_fold' + str(i + 1) + '_predictions.csv'), 
            #         p, delimiter=',', fmt='%.3f')


if __name__ == '__main__':
    main()
