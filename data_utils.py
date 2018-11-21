#!-*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
import h5py

# Define data tuple
Data = namedtuple('Data', ['X', 'y', 'time', 'Xf', 'ref_vals'])

def make_training_data(filename='data.h5', test_frac=0.2, in_seq_length=28,
                       out_seq_length=1, decoder=False, shuffle=True):
    """
    Create training and testing data from time series read in from HDF5 file.

    Args:
        filename: str
            Name of HDF5 file.
        test_frac: float
            Value between 0.0 and 1.0 representing fraction of examples to use for test set.
        in_seq_length: int
            Number of time steps of input sequence to RNN.
        out_seq_length: int
            Number of time steps of output sequence.
        decoder: bool
            Prepare data for training decoder in encoder-decoder architecture.
        shuffle: bool
            Shuffle examples before splitting into training and testing set.
    """
    # Cache sequence lengths
    T = in_seq_length
    F = out_seq_length

    # First load data from HDF5
    with h5py.File(filename, 'r') as fid:
        # Time array
        t = fid['t'].value
        # Time series values
        tseries = fid['ts'].value

    # Normalize time
    t -= t[0]
    print('Time span of sequence length:', t[T])

    # First create list of starting indices for each sequence
    n_data = t.size
    start_inds = np.arange(n_data - T - F, dtype=int)
    n_instances = len(start_inds)

    # Create arrays for data instances
    X = np.zeros((n_instances, T, 1), dtype=np.float32)
    y = np.zeros((n_instances, F, 1), dtype=np.float32)
    time = np.zeros((n_instances, T, 1), dtype=np.float32)
    ref_values = np.zeros(n_instances)

    # Check if decoder inputs have been requested
    if decoder:
        Xf = np.zeros((n_instances, F, 1), dtype=np.float32)

    # Fill them in
    for i in range(n_instances):

        # Fill in the input data to the encoder (subtract starting value)
        istart = start_inds[i]
        iend = istart + T
        ref_val = tseries[istart]
        ref_values[i] = ref_val
        X[i,:,0] = tseries[istart:iend] - ref_val
        time[i,:,0] = t[istart:iend]

        # Fill in the target output data
        istart = start_inds[i] + T
        iend = istart + F
        y[i,:,0] = tseries[istart:iend] - ref_val

        # Fill in input data to decoder if requested
        if decoder:
            istart = start_inds[i] + T - F
            iend = istart + F
            Xf[i,:,0] = tseries[istart:iend] - ref_val

    # Create random permutation indices for the instances
    if shuffle:
        ind_shuffle = np.random.permutation(n_instances)
    else:
        ind_shuffle = np.arange(n_instances, dtype=int)

    # Compute indices for splitting data into train and test sets
    n_train = int(n_instances * (1.0 - test_frac))
    ind_train = ind_shuffle[:n_train]
    ind_test = ind_shuffle[n_train:]

    # Split the data
    X_train, X_test = X[ind_train], X[ind_test]
    y_train, y_test = y[ind_train], y[ind_test]
    time_train, time_test = time[ind_train], time[ind_test]
    ref_vals_train, ref_vals_test = ref_values[ind_train], ref_values[ind_test]
    if decoder:
        Xf_train, Xf_test = Xf[ind_train], Xf[ind_test]
    else:
        Xf_train = Xf_test = None

    # Create data tuples and return them
    train = Data(X=X_train, y=y_train, time=time_train, Xf=Xf_train, ref_vals=ref_vals_train)
    test = Data(X=X_test, y=y_test, time=time_test, Xf=Xf_test, ref_vals=ref_vals_test)
    return train, test

# end of file
