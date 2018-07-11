import argparse
import operator
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data import get_batch
from meta_optimizer import FastMetaOptimizer, MetaModel, MetaOptimizer
from model import MnistModel, LinearRegressionModel
from utils import setup_data, setup_model, get_config


# PARSE INPUT
parser = argparse.ArgumentParser(description='PyTorch Meta Learning')
parser.add_argument('--problem', type=str, default='mnist', metavar='XXX',
                    help='select problem to learn (options: simple, mnist(default))')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 20)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--load_from_path', type=str, default='n', metavar='PATH',
                    help='load optimizer from specified path, loads latest if given "latest" (default: doesn\'t load)')
parser.add_argument('--optimizer_type', type=str, default='fast', metavar='OPT',
                    help='meta optimizer to use (options: fast (default), lstm)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0


def main():
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.

    train_loader, test_loader = setup_data(args)
    meta_model = setup_model(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    meta_model.to(device)

    optimizer_type = args.optimizer_type.lower()
    if optimizer_type == 'fast':
        meta_optimizer = FastMetaOptimizer(
            MetaModel(meta_model), args.num_layers, args.hidden_size).to(device)
    elif optimizer_type == 'lstm':
        meta_optimizer = MetaOptimizer(
            MetaModel(meta_model), args.num_layers, args.hidden_size).to(device)
    else:
        raise ValueError(
            "{} is not a valid optimizer type.".format(optimizer_type))

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)
        for _ in range(args.updates_per_epoch):
            # Sample a new model
            # if args.problem == 'mnist':
            #     model = MnistModel().to(device)
            # elif args.problem == 'simple':
            #     train_loader, test_loader = setup_data(args)
            #     model = 
            # else:
            #     raise ValueError(
            #         "{} is not a valid problem.".format(args.problem))
                    

            train_loader, test_loader = setup_data(args)
            model = setup_model(args)
            model.to(device)

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)


            # Compute initial loss of the model
            f_x = model(x)

            if args.problem == 'mnist':
                initial_loss = F.nll_loss(f_x, y)
            elif args.problem == 'simple':
                initial_loss = F.binary_cross_entropy_with_logits(
                    f_x, y.float())
            else:
                raise ValueError(
                    "{} is not a valid problem.".format(args.problem))

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                prev_loss = prev_loss.to(device)
                for _ in range(args.truncated_bptt_step):
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)

                    x, y = x.to(device), y.to(device)

                    # First we need to compute the gradients of the model
                    f_x = model(x)                  # forward propagation

                    # calculate loss
                    if args.problem == 'mnist':
                        loss = F.nll_loss(f_x, y)
                    elif args.problem == 'simple':
                        loss = F.binary_cross_entropy_with_logits(
                            f_x, y.float())
                    else:
                        raise ValueError(
                            "{} is not a valid problem.".format(args.problem))

                    model.zero_grad()               # set gradients of all params to zero
                    loss.backward()                 # calculate gradients

                    # Perfom a meta update using gradients from modelate using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)

                    if args.problem == 'mnist':
                        loss = F.nll_loss(f_x, y)
                    elif args.problem == 'simple':
                        loss = F.binary_cross_entropy_with_logits(
                            f_x, y.float())
                    else:
                        raise ValueError(
                            "{} is not a valid problem.".format(args.problem))

                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)

                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()
            print(loss.item())

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                                      decrease_in_loss / args.updates_per_epoch))


if __name__ == "__main__":
    main()
