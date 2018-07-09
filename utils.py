import math
import torch

import numpy as np
import matplotlib.pyplot as plt

range_of_means = 10.0
range_of_stds = 10.0


def generate_data_points(train_data_points, test_data_points, dim, parameters_pos, parameters_neg):
    """
    For a specific parameter set, samples points from two gaussians, 
    and assigns label 0 or 1 according to which one the data came from.
    """
    mean_pos, std_pos = parameters_pos
    mean_neg, std_neg = parameters_neg

    # use diagonal covariance
    cov_pos = std_pos**2 * np.eye(len(std_pos))
    cov_neg = std_neg**2 * np.eye(len(std_neg))

    size = (train_data_points + test_data_points) // 2

    data_pos = np.random.multivariate_normal(mean_pos, cov_pos, size)
    data_neg = np.random.multivariate_normal(mean_neg, cov_neg, size)

    data_pos = np.concatenate((data_pos, np.ones((size, 1))), axis=1)
    data_neg = np.concatenate((data_neg, np.zeros((size, 1))), axis=1)

    data = np.concatenate((data_pos, data_neg), axis=0)

    np.random.shuffle(data)

    return data[:train_data_points, :dim], data[:train_data_points, dim:], data[train_data_points:, :dim], data[train_data_points:, dim:]


def generate_data_sets(dim, n_obj_func, train_data_points, test_data_points):
    """
    Generates train and test dataset for @n_obj_func functions.
    Returns a list of tuples of (train_x, train_y, test_x, test_y).
    """
    # sample means of objective functions from 0-mean uniform random dist.
    means_pos = (np.random.rand(n_obj_func, dim) - 0.5) * range_of_means
    means_neg = (np.random.rand(n_obj_func, dim) - 0.5) * range_of_means
    stds_pos = np.random.rand(n_obj_func, dim) * range_of_stds
    stds_neg = np.random.rand(n_obj_func, dim) * range_of_stds

    data = list()
    for i in range(n_obj_func):
        data.append(generate_data_points(train_data_points, test_data_points, dim,
                                         (means_pos[i], stds_pos[i]), (means_neg[i], stds_neg[i])))

    return data


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1.view(-1,1), x2.view(-1,1)), 1)
