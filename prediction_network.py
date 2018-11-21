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

import data_utils

# Limit tensorflow to a single GPU on CUDA-enabled systems
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Experimental parameters
batch_size = 64
hidden = [128, 32]
epochs = 150

# Sequence length parameters for synthetic data (2-year time span)
T = 182     # Length of sequence input to encoder
F = 1       # Number of time steps to predict

def main():

    # Make training and testing data
    train, test = data_utils.make_training_data(
        in_seq_length=T, out_seq_length=F, filename='synthetic_data.h5', shuffle=False
    )
    n_train = train.X.shape[0]
    n_test = test.X.shape[0]

    # Compute the number of batches
    n_batches = math.ceil(n_train // batch_size)
    print('Number of training batches of size %d: %d' % (batch_size, n_batches))

    # Create placeholders
    X = tf.placeholder(tf.float32, [None, T, 1])
    y = tf.placeholder(tf.float32, [None, F, 1])
   
    # Generate graph for model output
    outputs = make_model(X)

    # Compute loss 
    loss = tf.reduce_mean(tf.square(outputs - y))

    # Make lists of all encoder and prediction net variables
    encoder_vars = []
    prediction_vars = []
    for var in tf.get_default_graph().get_collection('variables'):
        if var.name.startswith('rnn_encoder'):
            encoder_vars.append(var)
        elif var.name.startswith('prediction'):
            prediction_vars.append(var)

    # Select optimizer for prediction network variables only
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    train_op = optimizer.minimize(loss, var_list=prediction_vars)

    # Create savers for encoder and prediction net variables
    encoder_saver = tf.train.Saver(encoder_vars)
    prediction_saver = tf.train.Saver(prediction_vars)

    # Initialize session and graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    # Restore encoder variables
    encoder_saver.restore(sess, 'checkpoints/encoder/variables.cpkt')

    # Run training
    train_mse = []
    test_mse = []
    for epoch in range(epochs):

        # Get shuffling indices
        ind_shuffle = np.random.permutation(n_train)

        # Shuffle data
        X_train = train.X[ind_shuffle]
        y_train = train.y[ind_shuffle] 
       
        # Loop over batches
        start = 0
        for batch in range(n_batches):

            # Create feed dictionary for minibatch and update parameters
            bslice = slice(start, start + batch_size)
            feed_dict = {X: X_train[bslice], y: y_train[bslice]}
            sess.run(train_op, feed_dict=feed_dict)

            # Update batch index
            start += batch_size

        # Diagnostics: print MSE on full training and testing set
        if epoch % 1 == 0:

            # Make feed dicts with full data
            feed_dict_train = {X: train.X, y: train.y}
            feed_dict_test = {X: test.X, y: test.y}

            # Compute losses
            train_mse.append(sess.run(loss, feed_dict=feed_dict_train))
            test_mse.append(sess.run(loss, feed_dict=feed_dict_test))
            print('Epoch %4d train MSE: %12.8f   test MSE: %12.8f' % 
                 (epoch, train_mse[-1], test_mse[-1]))

    # Create a checkpoint to save results
    save_path = prediction_saver.save(sess, 'checkpoints/prediction/variables.ckpt')

    # Plot train and test errors
    fig1, ax1 = plt.subplots()
    ax1.plot(train_mse, label='Train MSE')
    ax1.plot(test_mse, label='Test MSE')
    leg = ax1.legend(loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')

    # Randomly select some test sequences to plot
    n_plot = 4
    ind = np.random.permutation(n_test)[:n_plot]
    X_plot = test.X[ind]
    y_plot = test.y[ind]

    # Predict outputs
    y_pred = sess.run(outputs, feed_dict={X: X_plot, y: y_plot})

    # Plot them
    fig2, axes = plt.subplots(nrows=n_plot)
    ind_x = np.arange(T)
    ind_y = np.arange(T, T + F)
    for i in range(n_plot):
        axes[i].plot(ind_x, X_plot[i,:,0].squeeze())
        axes[i].plot(ind_y, y_plot[i].squeeze(), 'o')
        axes[i].plot(ind_y, y_pred[i].squeeze(), 's')

    plt.show()


def make_model(X, layers=[128, 32]):
    """
    Generate model for feedforward prediction network on top of an RNN-encoder.

    Args:
        X: tf.Variable
            Input sequences for encoder.

    Returns:
        outputs: tf.Variable
            Graph for prediction network output.
    """
    
    # 2-layer LSTMCells for encoder network
    rnn_encoder_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in layers]
    rnn_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_encoder_layers)

    # Run encoder over inputs
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        rnn_encoder_cell, X, dtype=tf.float32, scope='rnn_encoder'
    )

    # Concatenate the final states of the encoder to form embedding vector
    embedding = tf.concat([encoder_state[0].h, encoder_state[1].h], axis=1)

    # Output dense layers with dropout in between
    activation = tf.nn.tanh
    with tf.variable_scope('prediction'):
        a = tf.layers.dense(embedding, 128, activation=activation)
        a = tf.layers.dropout(a, rate=0.1, training=True)
        a = tf.layers.dense(a, 64, activation=activation)
        a = tf.layers.dropout(a, rate=0.1, training=True)
        a = tf.layers.dense(a, 16, activation=activation)
        a = tf.layers.dropout(a, rate=0.1, training=True)
        out = tf.layers.dense(a, F, activation=None)

    # Finally, reshape outputs to match shape of target data
    outputs = tf.reshape(out, [-1, F, 1])
    
    return outputs


if __name__ == '__main__':
    main()

# end of file
