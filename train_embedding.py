#!/usr/bin/env python3.6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import h5py
import math
import sys
import os

# Limit tensorflow to a single GPU on CUDA-enabled systems
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Experimental parameters
batch_size = 64
hidden = [128, 32]

# Sequence length parameters for synthetic data (2-year time span)
T = 182     # Length of sequence input to encoder
F = 30      # Length of sequence input to decoder

def main():

    # Make training and testing data
    Xt_train, Xt_test, Xf_train, Xf_test, y_train, y_test, time_train, time_test = \
        make_training_data()
    n_train = Xt_train.shape[0]
    n_test = Xt_test.shape[0]

    # Compute the number of batches
    n_batches = math.ceil(n_train // batch_size)
    print('Number of training batches of size %d: %d' % (batch_size, n_batches))

    # Create placeholders
    Xt = tf.placeholder(tf.float32, [None, T, 1])
    Xf = tf.placeholder(tf.float32, [None, F, 1])
    y = tf.placeholder(tf.float32, [None, F, 1])
   
    # Generate graph for model output
    outputs, encoder_states = make_model(Xt, Xf)
    h0 = encoder_states[0].h
    h1 = encoder_states[1].h

    # Compute loss 
    loss = tf.reduce_mean(tf.square(outputs - y))

    # Select optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
    train_op = optimizer.minimize(loss)

    # Make lists of all encoder and decoder variables
    encoder_vars = []
    decoder_vars = []
    for var in tf.get_default_graph().get_collection('variables'):
        if var.name.startswith('rnn_encoder'):
            encoder_vars.append(var)
        else:
            decoder_vars.append(var)

    # Create savers for encoder and decoder variables
    encoder_saver = tf.train.Saver(encoder_vars)
    decoder_saver = tf.train.Saver(decoder_vars)
   
    # Initialize session and graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    # Run training
    epochs = 75
    train_mse = []
    test_mse = []
    for epoch in range(epochs):

        # Get shuffling indices
        ind_shuffle = np.random.permutation(n_train)

        # Shuffle data
        Xt_train = Xt_train[ind_shuffle]
        Xf_train = Xf_train[ind_shuffle]
        y_train = y_train[ind_shuffle] 
       
        # Loop over batches
        start = 0
        for batch in range(n_batches):

            # Create feed dictionary for minibatch and update parameters
            bslice = slice(start, start + batch_size)
            feed_dict = {Xf: Xf_train[bslice], Xt: Xt_train[bslice], y: y_train[bslice]}
            sess.run(train_op, feed_dict=feed_dict) 

        # Diagnostics: print MSE on full training and testing set
        if epoch % 1 == 0:

            # Make feed dicts with full data
            feed_dict_train = {Xf: Xf_train, Xt: Xt_train, y: y_train}
            feed_dict_test = {Xf: Xf_test, Xt: Xt_test, y: y_test}

            # Compute losses
            train_mse.append(sess.run(loss, feed_dict=feed_dict_train))
            test_mse.append(sess.run(loss, feed_dict=feed_dict_test))
            print('Epoch %4d train MSE: %12.8f   test MSE: %12.8f' % 
                 (epoch, train_mse[-1], test_mse[-1]))

            # Compute encoder embedding inputs over all training inputs
            h0_vals, h1_vals = sess.run([h0, h1], feed_dict={Xt: Xt_train})
            # Save to disk
            with h5py.File('epoch_embeddings/train_epoch-%03d.h5' % epoch, 'w') as fid:
                fid['h0'] = h0_vals
                fid['h1'] = h1_vals
                fid['t'] = time_train[ind_shuffle]
                
            # Compute encoder embedding inputs over all training inputs
            h0_vals, h1_vals = sess.run([h0, h1], feed_dict={Xt: Xt_test})
            # Save to disk
            with h5py.File('epoch_embeddings/test_epoch-%03d.h5' % epoch, 'w') as fid:
                fid['h0'] = h0_vals
                fid['h1'] = h1_vals
                fid['t'] = time_test

    # Create a checkpoint to save results
    save_path = encoder_saver.save(sess, 'checkpoints/encoder/variables.cpkt')
    save_path = decoder_saver.save(sess, 'checkpoints/decoder/variables.ckpt')

    # Plot train and test errors
    fig1, ax1 = plt.subplots()
    ax1.plot(train_mse, label='Train MSE')
    ax1.plot(test_mse, label='Test MSE')
    leg = ax1.legend(loc='best')

    # Randomly select some test sequences to plot
    n_plot = 4
    ind = np.random.permutation(n_test)[:n_plot]
    Xt_plot = Xt_test[ind]
    Xf_plot = Xf_test[ind]
    y_plot = y_test[ind]

    # Predict outputs
    y_pred = sess.run(outputs, feed_dict={Xt: Xt_plot, Xf: Xf_plot, y: y_plot})

    # Plot them
    fig2, axes = plt.subplots(nrows=n_plot)
    ind_x = np.arange(T)
    ind_y = np.arange(T, T + F)
    for i in range(n_plot):
        axes[i].plot(ind_x, Xt_plot[i,:,0].squeeze())
        axes[i].plot(ind_y, y_plot[i].squeeze())
        axes[i].plot(ind_y, y_pred[i].squeeze())

    plt.show()


def make_model(Xt, Xf, layers=[128, 32]):
    """
    Generate multi-layer LSTM encoder-decoder architecture.

    Args:
        Xt: tf.Variable
            Input sequences for encoder.
        Xf: tf.Variable
            Input sequences for decoder.

    Returns:
        outputs: tf.Variable
            Graph for decoder output.
        encoder_state: tf.Variable
            Hidden state of the encoder at the final time step of sequence.
    """
    
    # 2-layer LSTMCells for encoder network
    rnn_encoder_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in layers]
    rnn_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_encoder_layers)

    # 2-layer LSTMCells for decoder network (encoder in reverse)
    rnn_decoder_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in layers[::-1]]
    rnn_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_decoder_layers)

    # Run encoder over inputs
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        rnn_encoder_cell, Xt, dtype=tf.float32, scope='rnn_encoder'
    )

    # Run decoder over inputs and initializing with the encoder state
    decoder_outputs, decoder_state = tf.nn.dynamic_rnn(
        rnn_decoder_cell, Xf, initial_state=(encoder_state[1], encoder_state[0]),
        dtype=tf.float32, scope='rnn_decoder'
    )

    # Flatten-ish decoder output
    stacked_rnn_output = tf.reshape(decoder_outputs, [-1, layers[0]])

    # Output dense layer
    stacked_outputs = tf.layers.dense(stacked_rnn_output, 1)
    outputs = tf.reshape(stacked_outputs, [-1, F, 1])
    
    return outputs, encoder_state
  

def make_training_data(filename='data.h5', test_frac=0.2, shuffle=True):
    """
    Create training and testing data from time series read in from HDF5 file.

    Args:
        filename: str
            Name of HDF5 file.
        test_frac: float
            Value between 0.0 and 1.0 representing fraction of examples to use for test set.
        shuffle: bool
            Shuffle examples before splitting into training and testing set.
    """
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
    Xt = np.zeros((n_instances, T, 1), dtype=np.float32)
    Xf = np.zeros((n_instances, F, 1), dtype=np.float32)
    y = np.zeros((n_instances, F, 1), dtype=np.float32)
    time = np.zeros((n_instances, T, 1), dtype=np.float32)

    # Fill them in
    for i in range(n_instances):

        # Fill in the input data to the encoder (subtract starting value)
        istart = start_inds[i]
        iend = istart + T
        ref_val = tseries[istart]
        Xt[i,:,0] = tseries[istart:iend] - ref_val
        time[i,:,0] = t[istart:iend]

        # Fill in the input data to the decoder
        istart = start_inds[i] + T - F
        iend = istart + F
        Xf[i,:,0] = tseries[istart:iend] - ref_val

        # Fill in the target output data from the decoder
        istart = start_inds[i] + T
        iend = istart + F
        y[i,:,0] = tseries[istart:iend] - ref_val

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
    Xt_train, Xt_test = Xt[ind_train], Xt[ind_test]
    Xf_train, Xf_test = Xf[ind_train], Xf[ind_test]
    y_train, y_test = y[ind_train], y[ind_test]
    time_train, time_test = time[ind_train], time[ind_test]

    return Xt_train, Xt_test, Xf_train, Xf_test, y_train, y_test, time_train, time_test

 
if __name__ == '__main__':
    main()

# end of file