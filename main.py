import tensorflow as tf
from optimizers import SGD, RMSprop, RNNOptimizer
from problems import SimpleDG
from models.linear_regression_model import LinearRegressionModel
from trainers.linear_regression_trainer import LinearRegressionTrainer
from utils.dirs import create_dirs
from utils.logger import Logger
import matplotlib.pyplot as plt
import numpy as np


def learn(optimizer, model, n_steps):
    """Given an optimizer, this function returns a list of losses, which can be evaluated in a session.

    Arguments:
        optimizer {[type]} -- [description]
        model {[type]} -- [description]

    Returns:
        [list] -- List of tensors containing the losses. 
    """
    # create model
    x = model.create_problem(optimizer_name=optimizer.__class__.__name__)
    states = None
    for i in range(n_steps):
        # get loss
        loss = model.losses[i]
        # get gradients wrt loss
        grads = tf.gradients(loss, x)
        grads = [tf.stop_gradient(g) for g in grads]

        # create new model with updated weights
        updates, states = optimizer.step(grads, states)
        x = model.rollout(weights=x, updates=updates)

    return model.losses


def main():

    config = {
        "optimizer": "rnn",
        "training_steps": 20,  # This is 100 in the paper
        "learning_rate": 0.1,
        "decay_rate": 0.99,
        "layers": 2,
        "hidden_size": 20,
        "max_to_keep": 3,
        "dim": 10,
        "range_of_means": 10,
        "range_of_stds": 10,
        "summary_dir": "/summary",
        "checkpoint_dir": "/data_ckpt",
        "batch_size": 100
    }

    # create the experiments dirs
    create_dirs([config["summary_dir"], config["checkpoint_dir"]])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = SimpleDG(config)

    # create an instance of the model you want
    model = LinearRegressionModel(config)
    # create tensorboard logger
    # logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    # trainer = LinearRegressionTrainer(sess, model, data, config, logger)

    sess.run(tf.global_variables_initializer())

    if config["optimizer"] == "sgd":
        optim = SGD(config)
        losses = learn(optim, model, config["training_steps"])
    elif config["optimizer"] == "rms":
        optim = RMSprop(config)
        losses = learn(optim, model, config["training_steps"])
    elif config["optimizer"] == "rnn":
        optim = RNNOptimizer(config)
        losses = learn(optim, model, config["training_steps"])
        optim.train(losses, sess, data)

    x = np.arange(config["training_steps"] + 1)

    for _ in range(3):
        data_x, data_y = next(data.next_batch(config["batch_size"]))

        l = sess.run([losses], feed_dict={
                     "input:0": data_x, "label:0": data_y})
        print(l)

        p1, = plt.plot(x, l[0], label=config["optimizer"])
        plt.legend(handles=[p1])
        plt.title('Losses')
        plt.show()

        # TODO compare different optimizers

    # for _ in range(3):
    #     sgd_l, rms_l, rnn_l = sess.run(
    #         [sgd_losses, rms_losses, rnn_losses])
    #     p1, = plt.plot(x, sgd_l, label='SGD')
    #     p2, = plt.plot(x, rms_l, label='RMS')
    #     p3, = plt.plot(x, rnn_l, label='RNN')
    #     plt.legend(handles=[p1, p2, p3])
    #     plt.title('Losses')
    #     plt.show()

    #     p1, = plt.semilogy(x, sgd_l, label='SGD')
    #     p2, = plt.semilogy(x, rms_l, label='RMS')
    #     p3, = plt.semilogy(x, rnn_l, label='RNN')
    #     plt.legend(handles=[p1, p2, p3])
    #     plt.title('Losses')
    #     plt.show()


if __name__ == "__main__":
    main()
