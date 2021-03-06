import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class DataGenerator:
    def __init__(self, config):
        self.config = config

    def next_batch(self, batch_size):
        raise NotImplementedError


class SimpleDG(DataGenerator):
    def __init__(self, config):
        super(SimpleDG, self).__init__(config)
        self.dim = config["dim"]
        np.random.seed(0)

        self.refresh_parameters()

    def refresh_parameters(self, seed=None):
        np.random.seed(seed)
        self.mean_pos = (np.random.rand(self.dim) - 0.5) * \
            self.config['range_of_means']
        self.mean_neg = (np.random.rand(self.dim) - 0.5) * \
            self.config['range_of_means']
        std_pos = np.random.rand(self.dim) * self.config['range_of_stds']
        std_neg = np.random.rand(self.dim) * self.config['range_of_stds']

        # use diagonal covariance
        self.cov_pos = std_pos**2 * np.eye(self.dim)
        self.cov_neg = std_neg**2 * np.eye(self.dim)

    def next_batch(self, batch_size, mode=None):
        # TODO decide whether to always generate new examples or not

        pos_len = neg_len = batch_size // 2

        data_pos = np.random.multivariate_normal(
            self.mean_pos, self.cov_pos, pos_len)
        data_neg = np.random.multivariate_normal(
            self.mean_neg, self.cov_neg, neg_len)

        data_pos = np.concatenate(
            (data_pos, np.ones((pos_len, 1))), axis=1)
        data_neg = np.concatenate(
            (data_neg, np.zeros((neg_len, 1))), axis=1)

        self.data = np.concatenate((data_pos, data_neg), axis=0)

        np.random.shuffle(self.data)

        yield self.data[:, :-1], self.data[:, -1:]


class MNISTDG(DataGenerator):
    def __init__(self, config):
        super(MNISTDG, self).__init__(config)

        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def refresh_parameters(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def next_batch(self, batch_size, mode="train"):
        if mode == "train":
            indices = np.random.choice(
                self.data.train.num_examples, batch_size)
            batch_images = self.data.train.images[indices]
            batch_labels = self.data.train.labels[indices]
        elif mode == "test":
            indices = np.random.choice(self.data.test.num_examples, batch_size)
            batch_images = self.data.test.images[indices]
            batch_labels = self.data.test.labels[indices]

        yield batch_images, batch_labels
