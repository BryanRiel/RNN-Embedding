#!/usr/bin/env python3.6

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import tensorflow as tf
import h5py
import math
import sys
import os

# Limit tensorflow to a single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import data_utils
from train_embedding import T, F, make_model

def main():

    # Make training and testing data
    train, test = data_utils.make_training_data(
        in_seq_length=T, out_seq_length=F, filename='synthetic_data.h5',
        shuffle=False, decoder=True, test_frac=0.0
    )

    # Create placeholders for input data to encoder
    Xt = tf.placeholder(tf.float32, [None, T, 1])
    Xf = tf.placeholder(tf.float32, [None, F, 1])
   
    # Generate graph for model output
    outputs, encoder_states = make_model(Xt, Xf)
    h0 = encoder_states[0].h
    h1 = encoder_states[1].h

    # Create saver for variables
    saver = tf.train.Saver()
   
    # Initialize session and graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    # Make lists of encoder variables
    encoder_vars = []
    for var in tf.get_default_graph().get_collection('variables'):
        if var.name.startswith('rnn_encoder'):
            encoder_vars.append(var)

    # Create saver for encoder variables
    encoder_saver = tf.train.Saver(encoder_vars)

    # Restore variables
    encoder_saver.restore(sess, 'checkpoints/encoder/variables.cpkt')

    # Compute embedding inputs over all inputs
    h0_vals, h1_vals = sess.run([h0, h1], feed_dict={Xt: train.X})

    # Perform PCA on embeddings for visualization
    h0_vals = PCA(n_components=3, svd_solver='full').fit_transform(h0_vals)
    h1_vals = PCA(n_components=3, svd_solver='full').fit_transform(h1_vals)

    # Or use TSNE instead
    #h0_vals = TSNE(n_components=3, init='pca', n_iter=2000, 
    #               verbose=1).fit_transform(h0_vals)
    #h1_vals = TSNE(n_components=3, init='pca', n_iter=2000,
    #               verbose=1).fit_transform(h1_vals)

    # View embeddings in 3D
    fig = plt.figure(figsize=(12,4))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    #c = train.time[:,-1,:].squeeze() % 1.0
    c = train.time[:,-1,:].squeeze()
    sc0 = ax0.scatter(h0_vals[:,0], h0_vals[:,1], h0_vals[:,2], s=10, c=c, cmap='hsv')
    sc1 = ax1.scatter(h1_vals[:,0], h1_vals[:,1], h0_vals[:,2], s=10, c=c, cmap='hsv')

    plt.colorbar(sc0, ax=ax0)
    plt.colorbar(sc1, ax=ax1)

    plt.show()

 
if __name__ == '__main__':
    main()

# end of file
