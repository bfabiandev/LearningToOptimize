import numpy as np

n_obj_func = 90
n_data_points = 100
dim = 3
l2 = 0.0005
range_of_means = 10.0
range_of_stds = 10.0

def generate_data_points(number_of_data_points, parameters_pos, parameters_neg):
    mean_pos, std_pos = parameters_pos
    mean_neg, std_neg = parameters_neg
    
    # use diagonal covariance
    cov_pos = std_pos**2 * np.eye(len(std_pos))
    cov_neg = std_neg**2 * np.eye(len(std_neg))

    size = number_of_data_points // 2
    
    data_pos = np.random.multivariate_normal(mean_pos, cov_pos, size)
    data_neg = np.random.multivariate_normal(mean_neg, cov_neg, size)

    data_pos = np.concatenate((data_pos, np.ones((size, 1))), axis=1)
    data_neg = np.concatenate((data_neg, np.zeros((size, 1))), axis=1)

    data = np.concatenate((data_pos, data_neg), axis=0)
    
    np.random.shuffle(data)
    
    return data[:, :dim], data[:, dim:]
        

def generate_data_sets():
    # sample means of objective functions from 0-mean uniform random dist.
    means_pos = (np.random.rand(n_obj_func, dim) - 0.5) * range_of_means
    means_neg = (np.random.rand(n_obj_func, dim) - 0.5) * range_of_means
    stds_pos = np.random.rand(n_obj_func, dim) * range_of_stds
    stds_neg = np.random.rand(n_obj_func, dim) * range_of_stds

    data = list()
    for i in range(n_obj_func):
        data.append(generate_data_points(n_data_points, (means_pos[i], stds_pos[i]), (means_neg[i], stds_neg[i])))

    return data

generate_data_sets()