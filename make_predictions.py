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

import vanilla_lstm
import data_utils
from prediction_network import make_model, T, F

# Limit tensorflow to a single GPU on CUDA-enabled systems
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def main():

    # Make training and testing data
    train, test = data_utils.make_training_data(
        in_seq_length=T, out_seq_length=F, filename='synthetic_data.h5', shuffle=False
    )
    n_examples = test.X.shape[0]

    # Create placeholder for input data
    X = tf.placeholder(tf.float32, [None, T, 1])
   
    # Generate graph for model output
    outputs = make_model(X)

    # Generate graph for vanilla LSTM output
    vanilla_outputs = vanilla_lstm.make_model(X)

    # Make lists of all encoder and prediction net variables
    encoder_vars = []
    prediction_vars = []
    vanilla_vars = []
    for var in tf.get_default_graph().get_collection('variables'):
        if var.name.startswith('rnn_encoder'):
            encoder_vars.append(var)
        elif var.name.startswith('prediction'):
            prediction_vars.append(var)
        elif var.name.startswith('vanilla'):
            vanilla_vars.append(var)

    # Create savers
    encoder_saver = tf.train.Saver(encoder_vars)
    prediction_saver = tf.train.Saver(prediction_vars)
    vanilla_saver = tf.train.Saver(vanilla_vars)

    # Initialize session and graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    # Restore variables
    encoder_saver.restore(sess, 'checkpoints/encoder/variables.cpkt')
    prediction_saver.restore(sess, 'checkpoints/prediction/variables.ckpt')
    vanilla_saver.restore(sess, 'checkpoints/vanilla_lstm/variables.ckpt')

    t = np.zeros(n_examples)
    for i in range(n_examples):
        t[i] = test.time[i,-1,:]

    # Naive predictions
    y_naive = np.zeros(n_examples)
    for i in range(n_examples):
        y_naive[i] = test.X[i,-1,:].squeeze()
    y_naive += test.ref_vals

    # Vanilla LSTM predictions
    y_vanilla = sess.run(vanilla_outputs, feed_dict={X: test.X}).squeeze() + test.ref_vals

    # Initialize plots
    fig2, ax = plt.subplots()
    y_true = test.y.squeeze() + test.ref_vals
    ax.plot(t, y_true)

    # Run random trials for predictions
    n_trials = 100
    y_trials = np.zeros((n_trials, n_examples))
    for k in tqdm(range(n_trials)):
        y_trials[k,:] = sess.run(outputs, feed_dict={X: test.X}).squeeze()

    y_mean = np.mean(y_trials, axis=0) + test.ref_vals
    y_std = 2 * np.std(y_trials, axis=0)

    print('Naive MSE   :', np.mean((y_naive - y_true)**2))
    print('Vanilla MSE :', np.mean((y_vanilla - y_true)**2))
    print('PredNet MSE :', np.mean((y_mean - y_true)**2))

    line, = ax.plot(t, y_mean)
    ax.plot(t, y_naive)
    ax.plot(t, y_vanilla)
    ax.fill_between(t, y_mean - y_std, y_mean + y_std, color=line.get_color(), alpha=0.4)
    
    plt.show()


if __name__ == '__main__':
    main()

# end of file
