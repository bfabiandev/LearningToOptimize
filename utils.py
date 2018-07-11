import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from model import LinearRegressionModel, MnistModel
from torchvision import datasets, transforms


def get_config(problem):
    if problem == 'simple':
        return {
            'DATASET_LEN': 200,
            'TRAIN_PERCENT': 0.5,
            'RANGE_OF_MEANS': 10.0,
            'RANGE_OF_STDS': 10.0,
            'DIM': 3
        }
    else:
        raise ValueError("{} is not a valid problem".format(problem))


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def setup_data(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.problem == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.problem == 'simple':
        config = get_config(args.problem)
        # simple = Simple(length=config['DATASET_LEN'], dim=config['DIM'])

        # train_idx = np.random.choice(
        #     config['DATASET_LEN'], size=int(config['TRAIN_PERCENT'] * config['DATASET_LEN']), replace=False)
        # train_sampler = SubsetRandomSampler(train_idx)
        # train_loader = DataLoader(
        #     simple, batch_size=args.batch_size, sampler=train_sampler, shuffle=True)

        # test_idx = list(set(range(config['DATASET_LEN'])) - set(train_idx))
        # test_sampler = SubsetRandomSampler(test_idx)
        # test_loader = DataLoader(
        #     simple, batch_size=args.batch_size, sampler=test_sampler, shuffle=True)

        simple = Simple(length=config['DATASET_LEN'], dim=config['DIM'])
        train_loader = DataLoader(
            simple.train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            simple.test, batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError("{} is not a valid problem.".format(args.problem))

    return train_loader, test_loader


def setup_model(args):
    if args.problem == 'mnist':
        meta_model = MnistModel()
    elif args.problem == 'simple':
        config = get_config(args.problem)
        meta_model = LinearRegressionModel(dim=config['DIM'])
    else:
        raise ValueError("{} is not a valid problem.".format(args.problem))

    return meta_model


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1.view(-1, 1), x2.view(-1, 1)), 1)


def save(meta_optimizer, path):
    torch.save(meta_optimizer.state_dict(), path)


def load(meta_optimizer, path):
    meta_optimizer.load_state_dict(torch.load(path))
    return meta_optimizer


class Simple():
    def __init__(self, length=100, dim=3):
        self.len = length
        self.dim = dim

        config = get_config('simple')

        mean_pos = (np.random.rand(dim) - 0.5) * config['RANGE_OF_MEANS']
        mean_neg = (np.random.rand(dim) - 0.5) * config['RANGE_OF_MEANS']
        std_pos = np.random.rand(dim) * config['RANGE_OF_STDS']
        std_neg = np.random.rand(dim) * config['RANGE_OF_STDS']

        # use diagonal covariance
        cov_pos = std_pos**2 * np.eye(len(std_pos))
        cov_neg = std_neg**2 * np.eye(len(std_neg))

        class SimpleHelper(Dataset):
            def __init__(self, length=100, dim=3):
                self.len = length

                pos_len = neg_len = length // 2

                data_pos = np.random.multivariate_normal(
                    mean_pos, cov_pos, pos_len)
                data_neg = np.random.multivariate_normal(
                    mean_neg, cov_neg, neg_len)

                data_pos = np.concatenate(
                    (data_pos, np.ones((pos_len, 1))), axis=1)
                data_neg = np.concatenate(
                    (data_neg, np.zeros((neg_len, 1))), axis=1)

                self.data = np.concatenate((data_pos, data_neg), axis=0)

                np.random.shuffle(self.data)

            def __getitem__(self, index):
                return self.data[index, :-1], self.data[index, -1:]

            def __len__(self):
                return self.len

        self.train = SimpleHelper(length=self.len // 2, dim=self.dim)
        self.test = SimpleHelper(length=self.len // 2, dim=self.dim)
        