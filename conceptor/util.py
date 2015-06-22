"""
Created on May 25, 2015

@author: littleowen
@note: useful utilities functions for assisting conceptor networks
"""

import os;
import pickle as pickle
import numpy as np
import numpy.matlib
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from scipy.lib.six import xrange
from scipy.sparse.coo import coo_matrix


def sprandn(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    """Generate a sparse matrix of the given shape and density with standard
    normally distributed values.
    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str
        sparse matrix format.
    dtype : dtype
        type of the returned matrix values.
    random_state : {numpy.random.RandomState, int}, optional
        Random number generator or random seed. If not given, the singleton
        numpy.random will be used.
    Notes
    -----
    Only float types are supported for now.
    """
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")
    if dtype and (dtype not in [np.float32, np.float64, np.longdouble]):
        raise NotImplementedError("type %s not supported" % dtype)

    mn = m * n

    tp = np.intc
    if mn > np.iinfo(tp).max:
        tp = np.int64

    if mn > np.iinfo(tp).max:
        msg = """\
Trying to generate a random sparse matrix such as the product of dimensions is
greater than %d - this is not supported on this machine
"""
        raise ValueError(msg % np.iinfo(tp).max)

    # Number of non zero values
    k = int(density * m * n)

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, (int, np.integer)):
        random_state = np.random.RandomState(random_state)

    # Use the algorithm from python's random.sample for k < mn/3.
    if mn < 3*k:
        # We should use this line, but choice is only available in numpy >= 1.7
        # ind = random_state.choice(mn, size=k, replace=False)
        ind = random_state.permutation(mn)[:k]
    else:
        ind = np.empty(k, dtype=tp)
        selected = set()
        for i in xrange(k):
            j = random_state.randint(mn)
            while j in selected:
                j = random_state.randint(mn)
            selected.add(j)
            ind[i] = j

    j = np.floor(ind * 1. / m).astype(tp)
    i = (ind - j * m).astype(tp)
    vals = random_state.randn(k).astype(dtype)
    return coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format)

def generate_internal_weights(size_net,
                              density):
    """
    Generate internal weights in a reservoir

    @param size_net: number of neurons in the reservoir
    @param density: density of the network (connectivity)

    @return: weights: a sparse matrix of internal weights
    """
    success = 0
    while not success:
        try:
            weights = sprandn(m = size_net, n = size_net, density = density, format = 'coo')
            eigw, _ = scipy.sparse.linalg.eigs(weights, 1)
            success = 1
        except:
            pass
    weights /= np.abs(eigw[0])

    return weights.toarray()

def init_weights(size_in,
                 size_net,
                 sr,
                 in_scale,
                 bias_scale):
    """
    Initialize weights for a new conceptor network

    @param size_in: number of input
    @param size_net: number of internal neurons 
    @param sr: spectral radius
    @param in_scale: scaling of input weights
    @param bias_scale: size of bias
    """

    # generate internal weights
    if size_net <= 20:
        connectivity = 1
    else:
        connectivity = 10. / size_net
    W_star_raw = generate_internal_weights(size_net = size_net, density = connectivity)

    W_star = W_star_raw * sr

    # generate input weights
    W_in = np.random.randn(size_net, size_in) * in_scale

    # generate bias
    W_bias = np.random.randn(size_net, 1) * bias_scale

    return W_star, W_in, W_bias

def consecdata(datavec, timestep = 4):
    resultvec = datavec
    for i in range(timestep):
        resultvec = np.dstack([resultvec, np.hstack([datavec[:, (i + 1):], datavec[:, 0:(i + 1)]])])
    return resultvec


def normalize_data(data):
    """
    Nomarlize all the training data to be in range [0,1]   

    @param data: a list of training data, each corresponding to one class to be recognized
    """

    num_data = len(data)
    all_data = np.hstack(data)

    max_vals = np.max(all_data, 1)
    min_vals = np.min(all_data, 1)
    shifts = - min_vals
    scales = 1. / (max_vals - min_vals)
    norm_data = []

    for s in range(num_data):
        d = data[s] + np.matlib.repmat(shifts[None].T, 1 ,data[s].shape[1])
        d = d.T.dot(np.diag(scales)).T
        norm_data.append(d)
    return norm_data, shifts, scales



def transform_data(data, shifts, scales):
    num_data = len(data)
    trans_data = []
    for s in range(num_data):
        d = data[s] + np.matlib.repmat(shifts[None].T, 1, data[s].shape[1])
        d = d.T.dot(np.diag(scales)).T
        trans_data.append(d)
    return trans_data

